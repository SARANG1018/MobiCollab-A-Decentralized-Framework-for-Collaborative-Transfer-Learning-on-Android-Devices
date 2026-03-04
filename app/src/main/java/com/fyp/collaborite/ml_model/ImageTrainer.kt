/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.modelpersonalization

import android.content.Context
import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min


class TransferLearningHelper(
    var numThreads: Int = 2,
    val context: Context,
    val classifierListener: ClassifierListener?
) {

    private var interpreter: Interpreter? = null
    private val trainingSamples: MutableList<TrainingSample> = mutableListOf()
    private var executor: ExecutorService? = null
    public fun getSampleCount(): Int {
        return trainingSamples.size
    }

    private val lock = Any()
    private var targetWidth: Int = 0
    private var targetHeight: Int = 0
    private val handler = Handler(Looper.getMainLooper())

    init {
        if (setupModelPersonalization()) {
            targetWidth = interpreter!!.getInputTensor(0).shape()[2]
            targetHeight = interpreter!!.getInputTensor(0).shape()[1]
        } else {
            classifierListener?.onError("TFLite failed to init.")
        }
    }

    fun close() {
        executor?.shutdownNow()
        executor = null
        interpreter = null
    }

    fun pauseTraining() {
        executor?.shutdownNow()
    }

    // Expose interpreter for tensor access (used in federated averaging)
    fun getInterpreter(): Interpreter? {
        return interpreter
    }

    // Get information about all tensors in the model
    fun inspectModelTensors(): String {
        synchronized(lock) {
            if (interpreter == null) {
                setupModelPersonalization()
            }

            val sb = StringBuilder()
            sb.appendLine("=== MODEL TENSOR INSPECTION (ENHANCED) ===")

            try {
                // Input tensors
                val inputCount = interpreter?.inputTensorCount ?: 0
                sb.appendLine("\nInput Tensors: $inputCount")
                for (i in 0 until inputCount) {
                    val tensor = interpreter?.getInputTensor(i)
                    sb.appendLine("  [$i] Shape: ${tensor?.shape()?.contentToString()}, " +
                            "Type: ${tensor?.dataType()}, " +
                            "Elements: ${tensor?.numElements()}")
                }

                // Output tensors
                val outputCount = interpreter?.outputTensorCount ?: 0
                sb.appendLine("\nOutput Tensors: $outputCount")
                for (i in 0 until outputCount) {
                    val tensor = interpreter?.getOutputTensor(i)
                    sb.appendLine("  [$i] Shape: ${tensor?.shape()?.contentToString()}, " +
                            "Type: ${tensor?.dataType()}, " +
                            "Elements: ${tensor?.numElements()}")
                }

                // Try to access signature-specific tensors
                sb.appendLine("\n--- Checking Signature Tensors ---")
                try {
                    val signatureKeys = interpreter?.signatureKeys
                    sb.appendLine("Available signatures: ${signatureKeys?.joinToString(", ")}")

                    // Check save signature outputs
                    signatureKeys?.forEach { sigKey ->
                        try {
                            val inputs = interpreter?.getSignatureInputs(sigKey)
                            val outputs = interpreter?.getSignatureOutputs(sigKey)
                            sb.appendLine("\nSignature '$sigKey':")
                            sb.appendLine("  Inputs: ${inputs?.joinToString(", ")}")
                            sb.appendLine("  Outputs: ${outputs?.joinToString(", ")}")
                        } catch (e: Exception) {
                            sb.appendLine("  Error inspecting signature '$sigKey': ${e.message}")
                        }
                    }
                } catch (e: Exception) {
                    sb.appendLine("Error accessing signatures: ${e.message}")
                }

                sb.appendLine("\n=== END INSPECTION ===")
            } catch (e: Exception) {
                sb.appendLine("Error inspecting tensors: ${e.message}")
                Log.e(TAG, "Error in inspectModelTensors", e)
            }

            return sb.toString()
        }
    }

    fun extractWeights(checkpointDir: String): ModelWeights? {
        synchronized(lock) {
            if (interpreter == null) {
                setupModelPersonalization()
            }

            return try {
                val checkpointPath = "$checkpointDir/temp_extract.ckpt"
                saveWeightsInternal(checkpointPath)

                val wsOutput = Array(BOTTLENECK_SIZE) { FloatArray(NUM_CLASSES) }
                val bsOutput = Array(1) { FloatArray(NUM_CLASSES) }

                val inputs = HashMap<String, Any>()
                inputs["checkpoint_path"] = checkpointPath
                val outputs = HashMap<String, Any>()
                outputs["ws"] = wsOutput
                outputs["bs"] = bsOutput

                interpreter?.runSignature(inputs, outputs, "restore")

                Log.d(TAG, "[EXTRACT] Weights extracted. ws[0][0..3]=${wsOutput[0].take(4)}, bs=${bsOutput[0].toList()}")
                ModelWeights(ws = wsOutput, bs = bsOutput[0])
            } catch (e: Exception) {
                Log.e(TAG, "Failed to extract weights: ${e.message}", e)
                null
            }
        }
    }

    fun setWeights(weights: ModelWeights, checkpointDir: String): Boolean {
        synchronized(lock) {
            if (interpreter == null) setupModelPersonalization()

            try {
                val checkpointPath = "$checkpointDir/fedavg_import.ckpt"
                saveWeightsInternal(checkpointPath)

                val file = File(checkpointPath)
                val fileBytes = file.readBytes()
                val buf = ByteBuffer.wrap(fileBytes)
                buf.order(ByteOrder.LITTLE_ENDIAN)

                buf.position(CKPT_BS_OFFSET)
                for (v in weights.bs) buf.putFloat(v)

                buf.position(CKPT_WS_OFFSET)
                for (row in weights.ws) {
                    for (v in row) buf.putFloat(v)
                }

                val rawCrc1 = crc32c(fileBytes, 0, CKPT_BLOCK1_CRC_OFFSET)
                val masked1 = ((rawCrc1 ushr 15) or (rawCrc1 shl 17)) + CRC_MASK_DELTA
                buf.position(CKPT_BLOCK1_CRC_OFFSET)
                buf.putInt(masked1)

                val rawCrc2 = crc32c(fileBytes, CKPT_BLOCK2_START, CKPT_BLOCK2_CRC_OFFSET)
                val masked2 = ((rawCrc2 ushr 15) or (rawCrc2 shl 17)) + CRC_MASK_DELTA
                buf.position(CKPT_BLOCK2_CRC_OFFSET)
                buf.putInt(masked2)

                file.writeBytes(fileBytes)

                val inputs = HashMap<String, Any>()
                inputs["checkpoint_path"] = checkpointPath
                interpreter?.runSignature(inputs, HashMap(), "restore")
                Log.d(TAG, "[INJECT] Averaged weights injected into model")
                return true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to set weights: ${e.message}", e)
                return false
            }
        }
    }

    private fun saveWeightsInternal(checkpointPath: String) {
        val inputs = HashMap<String, Any>()
        inputs["checkpoint_path"] = checkpointPath
        interpreter?.runSignature(inputs, HashMap(), "save")
    }

    /**
     * Reset the model to fresh random weights by running the 'initialize' signature.
     * Use this when starting a new training session with different class meanings.
     * 
     * @param checkpointDir Optional directory to delete old checkpoint files from
     * @return true if reset succeeded
     */
    fun resetCheckpoint(checkpointDir: String? = null): Boolean {
        synchronized(lock) {
            if (interpreter == null) {
                setupModelPersonalization()
            }

            return try {
                if (checkpointDir != null) {
                    val dir = File(checkpointDir)
                    if (dir.exists() && dir.isDirectory) {
                        dir.listFiles()?.filter { it.name.endsWith(".ckpt") }?.forEach { 
                            it.delete() 
                        }
                    }
                }

                val wsOutput = Array(BOTTLENECK_SIZE) { FloatArray(NUM_CLASSES) }
                val bsOutput = FloatArray(NUM_CLASSES)

                val inputs = HashMap<String, Any>()
                val outputs = HashMap<String, Any>()
                outputs["ws"] = wsOutput
                outputs["bs"] = bsOutput

                interpreter?.runSignature(inputs, outputs, "initialize")

                trainingSamples.clear()

                Log.i(TAG, "Model reset to fresh random weights")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Failed to reset checkpoint: ${e.message}", e)
                false
            }
        }
    }

    /**
     * Perform Federated Averaging on multiple sets of model weights.
     * Computes a weighted average based on the number of samples each client used for training.
     * 
     * @param weightsList List of ModelWeights from different clients/devices
     * @param sampleCounts Number of training samples used by each client (for weighted averaging)
     * @return Averaged ModelWeights, or null if inputs are invalid
     */
    fun performFederatedAveraging(
        weightsList: List<ModelWeights>,
        sampleCounts: List<Int>
    ): ModelWeights? {
        if (weightsList.isEmpty()) {
            Log.e(TAG, "FedAvg: Empty weights list")
            return null
        }

        if (weightsList.size != sampleCounts.size) {
            Log.e(TAG, "FedAvg: weightsList.size (${weightsList.size}) != sampleCounts.size (${sampleCounts.size})")
            return null
        }

        if (sampleCounts.any { it <= 0 }) {
            Log.e(TAG, "FedAvg: All sample counts must be positive")
            return null
        }

        val totalSamples = sampleCounts.sum().toFloat()
        val numClients = weightsList.size

        val wsRows = weightsList[0].ws.size
        val wsCols = weightsList[0].ws[0].size
        val bsSize = weightsList[0].bs.size

        for (i in weightsList.indices) {
            if (weightsList[i].ws.size != wsRows || 
                weightsList[i].ws[0].size != wsCols ||
                weightsList[i].bs.size != bsSize) {
                Log.e(TAG, "FedAvg: Client $i has mismatched weight dimensions")
                return null
            }
        }

        // Compute weighted average for ws
        val avgWs = Array(wsRows) { FloatArray(wsCols) }
        for (row in 0 until wsRows) {
            for (col in 0 until wsCols) {
                var sum = 0f
                for (client in 0 until numClients) {
                    val weight = sampleCounts[client].toFloat() / totalSamples
                    sum += weightsList[client].ws[row][col] * weight
                }
                avgWs[row][col] = sum
            }
        }

        // Compute weighted average for bs
        val avgBs = FloatArray(bsSize)
        for (i in 0 until bsSize) {
            var sum = 0f
            for (client in 0 until numClients) {
                val weight = sampleCounts[client].toFloat() / totalSamples
                sum += weightsList[client].bs[i] * weight
            }
            avgBs[i] = sum
        }

        Log.i(TAG, "FedAvg completed: $numClients clients, $totalSamples total samples")
        return ModelWeights(ws = avgWs, bs = avgBs)
    }

    private fun setupModelPersonalization(): Boolean {
        val options = Interpreter.Options()
        options.numThreads = numThreads
        return try {
            val modelFile = FileUtil.loadMappedFile(context, "model.tflite")
            interpreter = Interpreter(modelFile, options)
            true
        } catch (e: IOException) {
            classifierListener?.onError(
                "Model personalization failed to " +
                        "initialize. See error logs for details"
            )
            Log.e(TAG, "TFLite failed to load model with error: " + e.message)
            false
        }
    }

    // Process input image and add the output into list samples which are
    // ready for training.
    fun addSample(image: Bitmap, className: String, rotation: Int) {
        synchronized(lock) {
            if (interpreter == null) {
                setupModelPersonalization()
            }
            processInputImage(image, rotation)?.let { tensorImage ->
                val bottleneck = loadBottleneck(tensorImage)
                trainingSamples.add(
                    TrainingSample(
                        bottleneck,
                        encoding(classes.getValue(className))
                    )
                )
                Log.d(TAG, "[SAMPLE] Added class=$className, total=${trainingSamples.size}, bottleneck[0..2]=${bottleneck.take(3)}")
            }
        }
    }

    // Start training process
    fun startTraining() {
        if (interpreter == null) {
            setupModelPersonalization()
        }

        // Create new thread for training process.
        executor = Executors.newSingleThreadExecutor()
        val trainBatchSize = getTrainBatchSize()

        if (trainingSamples.size < trainBatchSize) {
            throw RuntimeException(
                String.format(
                    "Too few samples to start training: need %d, got %d",
                    trainBatchSize, trainingSamples.size
                )
            )
        }

        executor?.execute {
            synchronized(lock) {
                var avgLoss: Float

                // Keep training until the helper pause or close.
                while (executor?.isShutdown == false) {
                    var totalLoss = 0f
                    var numBatchesProcessed = 0

                    // Shuffle training samples to reduce overfitting and
                    // variance.
                    trainingSamples.shuffle()

                    trainingBatches(trainBatchSize)
                        .forEach { trainingSamples ->
                            val trainingBatchBottlenecks =
                                MutableList(trainBatchSize) {
                                    FloatArray(
                                        BOTTLENECK_SIZE
                                    )
                                }

                            val trainingBatchLabels =
                                MutableList(trainBatchSize) {
                                    FloatArray(
                                        classes.size
                                    )
                                }

                            // Copy a training sample list into two different
                            // input training lists.
                            trainingSamples.forEachIndexed { index, trainingSample ->
                                trainingBatchBottlenecks[index] =
                                    trainingSample.bottleneck
                                trainingBatchLabels[index] =
                                    trainingSample.label
                            }

                            val loss = training(
                                trainingBatchBottlenecks,
                                trainingBatchLabels
                            )
                            totalLoss += loss
                            numBatchesProcessed++
                        }

                    // Calculate the average loss after training all batches.
                    avgLoss = totalLoss / numBatchesProcessed
                    handler.post {
                        classifierListener?.onLossResults(avgLoss)
                    }
                }
            }
        }
    }

    private fun training(
        bottlenecks: MutableList<FloatArray>,
        labels: MutableList<FloatArray>
    ): Float {
        val inputs: MutableMap<String, Any> = HashMap()
        inputs[TRAINING_INPUT_BOTTLENECK_KEY] = bottlenecks.toTypedArray()
        inputs[TRAINING_INPUT_LABELS_KEY] = labels.toTypedArray()

        val outputs: MutableMap<String, Any> = HashMap()
        val loss = FloatBuffer.allocate(1)
        outputs[TRAINING_OUTPUT_KEY] = loss

        interpreter?.runSignature(inputs, outputs, TRAINING_KEY)
        return loss.get(0)
    }

    // Save trainable weights to checkpoint file
    fun saveWeights(checkpointPath: String): Boolean {
        synchronized(lock) {
            if (interpreter == null) {
                setupModelPersonalization()
            }

            return try {
                val inputs: MutableMap<String, Any> = HashMap()
                inputs["checkpoint_path"] = checkpointPath
                val outputs: MutableMap<String, Any> = HashMap()

                interpreter?.runSignature(inputs, outputs, "save")
                Log.d(TAG, "[SAVE] Weights saved to $checkpointPath")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Error saving weights: ${e.message}", e)
                false
            }
        }
    }

    // Restore trainable weights from checkpoint file
    fun restoreWeights(checkpointPath: String): Boolean {
        synchronized(lock) {
            if (interpreter == null) {
                setupModelPersonalization()
            }

            return try {
                val inputs: MutableMap<String, Any> = HashMap()
                inputs["checkpoint_path"] = checkpointPath
                val outputs: MutableMap<String, Any> = HashMap()

                interpreter?.runSignature(inputs, outputs, "restore")
                Log.d(TAG, "[RESTORE] Weights restored from $checkpointPath")
                true
            } catch (e: Exception) {
                Log.e(TAG, "Error restoring weights: ${e.message}", e)
                false
            }
        }
    }

    // Invokes inference on the given image batches.
    fun classify(bitmap: Bitmap, rotation: Int) {
        Log.d(TAG, "[CLASSIFY] Starting inference...")
        processInputImage(bitmap, rotation)?.let { image ->
            synchronized(lock) {
                if (interpreter == null) {
                    setupModelPersonalization()
                }

                // Inference time is the difference between the system time at the start and finish of the
                // process
                var inferenceTime = SystemClock.uptimeMillis()

                val inputs: MutableMap<String, Any> = HashMap()
                inputs[INFERENCE_INPUT_KEY] = image.buffer

                val outputs: MutableMap<String, Any> = HashMap()
                val output = TensorBuffer.createFixedSize(
                    intArrayOf(1, 4),
                    DataType.FLOAT32
                )
                outputs[INFERENCE_OUTPUT_KEY] = output.buffer

                interpreter?.runSignature(inputs, outputs, INFERENCE_KEY)
                val tensorLabel = TensorLabel(classes.keys.toList(), output)
                val result = tensorLabel.categoryList

                inferenceTime = SystemClock.uptimeMillis() - inferenceTime

                classifierListener?.onResults(result, inferenceTime)
            }
        }
    }

    // Loads the bottleneck feature from the given image array.
    private fun loadBottleneck(image: TensorImage): FloatArray {
        val inputs: MutableMap<String, Any> = HashMap()
        inputs[LOAD_BOTTLENECK_INPUT_KEY] = image.buffer
        val outputs: MutableMap<String, Any> = HashMap()
        val bottleneck = Array(1) { FloatArray(BOTTLENECK_SIZE) }
        outputs[LOAD_BOTTLENECK_OUTPUT_KEY] = bottleneck
        interpreter?.runSignature(inputs, outputs, LOAD_BOTTLENECK_KEY)
        return bottleneck[0]
    }

    // Preprocess the image and convert it into a TensorImage for classification.
    private fun processInputImage(
        image: Bitmap,
        imageRotation: Int
    ): TensorImage? {
        val height = image.height
        val width = image.width
        val cropSize = min(height, width)
        val imageProcessor = ImageProcessor.Builder()
            .add(Rot90Op(-imageRotation / 90))
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(
                ResizeOp(
                    targetHeight,
                    targetWidth,
                    ResizeOp.ResizeMethod.BILINEAR
                )
            )
            .add(NormalizeOp(0f, 255f))
            .build()
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(image)
        return imageProcessor.process(tensorImage)
    }

    // encode the classes name to float array
    private fun encoding(id: Int): FloatArray {
        val classEncoded = FloatArray(4) { 0f }
        classEncoded[id] = 1f
        return classEncoded
    }

    // Training model expected batch size.
    private fun getTrainBatchSize(): Int {
        return min(
            max( /* at least one sample needed */1, trainingSamples.size),
            EXPECTED_BATCH_SIZE
        )
    }

    // Constructs an iterator that iterates over training sample batches.
    private fun trainingBatches(trainBatchSize: Int): Iterator<List<TrainingSample>> {
        return object : Iterator<List<TrainingSample>> {
            private var nextIndex = 0

            override fun hasNext(): Boolean {
                return nextIndex < trainingSamples.size
            }

            override fun next(): List<TrainingSample> {
                val fromIndex = nextIndex
                val toIndex: Int = nextIndex + trainBatchSize
                nextIndex = toIndex
                return if (toIndex >= trainingSamples.size) {
                    // To keep batch size consistent, last batch may include some elements from the
                    // next-to-last batch.
                    trainingSamples.subList(
                        trainingSamples.size - trainBatchSize,
                        trainingSamples.size
                    )
                } else {
                    trainingSamples.subList(fromIndex, toIndex)
                }
            }
        }
    }

    interface ClassifierListener {
        fun onError(error: String)
        fun onResults(results: List<Category>?, inferenceTime: Long)
        fun onLossResults(lossNumber: Float)
    }

    companion object {
        const val CLASS_ONE = "1"
        const val CLASS_TWO = "2"
        const val CLASS_THREE = "3"
        const val CLASS_FOUR = "4"
        private val classes = mapOf(
            CLASS_ONE to 0,
            CLASS_TWO to 1,
            CLASS_THREE to 2,
            CLASS_FOUR to 3
        )
        private const val LOAD_BOTTLENECK_INPUT_KEY = "feature"
        private const val LOAD_BOTTLENECK_OUTPUT_KEY = "bottleneck"
        private const val LOAD_BOTTLENECK_KEY = "load"

        private const val TRAINING_INPUT_BOTTLENECK_KEY = "bottleneck"
        private const val TRAINING_INPUT_LABELS_KEY = "label"
        private const val TRAINING_OUTPUT_KEY = "loss"
        private const val TRAINING_KEY = "train"

        private const val INFERENCE_INPUT_KEY = "feature"
        private const val INFERENCE_OUTPUT_KEY = "output"
        private const val INFERENCE_KEY = "infer"

        private const val BOTTLENECK_SIZE = 1 * 7 * 7 * 1280
        private const val NUM_CLASSES = 4
        private const val EXPECTED_BATCH_SIZE = 20
        private const val TAG = "ModelPersonalizationHelper"

        private const val CKPT_BS_OFFSET = 97
        private const val CKPT_WS_OFFSET = 168
        private const val CKPT_BLOCK1_CRC_OFFSET = 122
        private const val CKPT_BLOCK2_START = 126
        private const val CKPT_BLOCK2_CRC_OFFSET = 1003697
        private const val CRC_MASK_DELTA = 0xa282ead8.toInt()
        private const val CRC32C_POLY = 0x82F63B78.toInt()

        fun crc32c(data: ByteArray, off: Int, len: Int): Int {
            var crc = 0.inv()
            for (i in off until len) {
                crc = crc xor (data[i].toInt() and 0xFF)
                for (j in 0 until 8) {
                    crc = if (crc and 1 != 0) {
                        (crc ushr 1) xor CRC32C_POLY
                    } else {
                        crc ushr 1
                    }
                }
            }
            return crc.inv()
        }
    }

    data class TrainingSample(val bottleneck: FloatArray, val label: FloatArray)

    data class ModelWeights(
        val ws: Array<FloatArray>,
        val bs: FloatArray
    )
}
