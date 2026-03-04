package com.fyp.collaborite.tests

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.examples.modelpersonalization.TransferLearningHelper
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Phase 3 Preparation: Model Tensor Inspection Test
 *
 * This test inspects the TFLite model to identify:
 * 1. Input and output tensor indices
 * 2. Tensor shapes and data types
 * 3. Which tensors contain trainable weights (ws, bs)
 *
 * KEY FINDING: The 'initialize' and 'restore' signatures
 * output 'ws' (weights) and 'bs' (biases) tensors!
 */
@RunWith(AndroidJUnit4::class)
class ModelTensorInspectionTest {

    private lateinit var context: Context
    private lateinit var imageTrainer: TransferLearningHelper

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        imageTrainer = TransferLearningHelper(
            context = context,
            classifierListener = null
        )
    }

    @Test
    fun testInspectModelTensors() {
        println("\n" + "=".repeat(80))
        println("MODEL TENSOR INSPECTION TEST")
        println("=".repeat(80))

        val tensorInfo = imageTrainer.inspectModelTensors()
        println(tensorInfo)
        Log.i("TensorInspection", tensorInfo)

        println("=".repeat(80))
    }

    @Test
    fun testReadWeightTensors() {
        println("\n" + "=".repeat(80))
        println("WEIGHT TENSOR EXTRACTION TEST")
        println("=".repeat(80))

        val interpreter = imageTrainer.getInterpreter()
        if (interpreter == null) {
            println("ERROR: Interpreter is null!")
            return
        }

        // --- Test 1: Read weights from 'initialize' signature ---
        println("\n--- Testing 'initialize' signature ---")
        try {
            // Get the output tensor info BEFORE running
            val wsOutputTensor = interpreter.getOutputTensorFromSignature("ws", "initialize")
            val bsOutputTensor = interpreter.getOutputTensorFromSignature("bs", "initialize")

            println("BEFORE run:")
            println("  ws tensor: shape=${wsOutputTensor.shape().contentToString()}, " +
                    "type=${wsOutputTensor.dataType()}, " +
                    "elements=${wsOutputTensor.numElements()}, " +
                    "bytes=${wsOutputTensor.numBytes()}")
            println("  bs tensor: shape=${bsOutputTensor.shape().contentToString()}, " +
                    "type=${bsOutputTensor.dataType()}, " +
                    "elements=${bsOutputTensor.numElements()}, " +
                    "bytes=${bsOutputTensor.numBytes()}")
            Log.i("TensorInspection", "ws shape: ${wsOutputTensor.shape().contentToString()}, elements: ${wsOutputTensor.numElements()}")
            Log.i("TensorInspection", "bs shape: ${bsOutputTensor.shape().contentToString()}, elements: ${bsOutputTensor.numElements()}")

            // Allocate output buffers based on tensor shapes
            val wsShape = wsOutputTensor.shape()
            val bsShape = bsOutputTensor.shape()

            val wsSize = wsOutputTensor.numElements()
            val bsSize = bsOutputTensor.numElements()

            // Create output arrays matching the shape
            val wsOutput: Any = if (wsShape.size == 2) {
                Array(wsShape[0]) { FloatArray(wsShape[1]) }
            } else {
                FloatArray(wsSize)
            }

            val bsOutput: Any = if (bsShape.size == 1) {
                FloatArray(bsSize)
            } else {
                Array(bsShape[0]) { FloatArray(bsShape[1]) }
            }

            // Run initialize signature
            val inputs = HashMap<String, Any>()
            val outputs = HashMap<String, Any>()
            outputs["ws"] = wsOutput
            outputs["bs"] = bsOutput

            interpreter.runSignature(inputs, outputs, "initialize")

            println("\nAFTER 'initialize' run:")
            println("  Total trainable parameters: ws=$wsSize + bs=$bsSize = ${wsSize + bsSize}")
            println("  Total bytes: ${(wsSize + bsSize) * 4}")

            // Print first few weight values to verify they're real numbers
            if (wsOutput is Array<*> && wsOutput.isNotEmpty()) {
                val firstRow = wsOutput[0] as FloatArray
                println("  ws first 5 values: ${firstRow.take(5)}")
                println("  ws last 5 values: ${firstRow.takeLast(5)}")
            } else if (wsOutput is FloatArray) {
                println("  ws first 5 values: ${wsOutput.take(5)}")
            }

            if (bsOutput is FloatArray) {
                println("  bs values: ${bsOutput.toList()}")
            }

            Log.i("TensorInspection", "SUCCESS: ws=$wsSize params, bs=$bsSize params, total=${wsSize + bsSize}")

        } catch (e: Exception) {
            println("ERROR with 'initialize' signature: ${e.message}")
            Log.e("TensorInspection", "Error reading initialize", e)
            e.printStackTrace()
        }

        // --- Test 2: Read weights from 'restore' signature ---
        println("\n--- Testing 'restore' signature ---")
        try {
            // First save a checkpoint
            val checkpointDir = context.filesDir.resolve("federated_checkpoints")
            checkpointDir.mkdirs()
            val checkpointPath = checkpointDir.resolve("weight_read_test.ckpt").absolutePath

            imageTrainer.saveWeights(checkpointPath)
            println("Saved checkpoint: ${java.io.File(checkpointPath).length()} bytes")

            // Get tensor info for restore outputs
            val wsOutputTensor = interpreter.getOutputTensorFromSignature("ws", "restore")
            val bsOutputTensor = interpreter.getOutputTensorFromSignature("bs", "restore")

            val wsShape = wsOutputTensor.shape()
            val bsShape = bsOutputTensor.shape()
            val wsSize = wsOutputTensor.numElements()
            val bsSize = bsOutputTensor.numElements()

            println("  restore ws: shape=${wsShape.contentToString()}, elements=$wsSize")
            println("  restore bs: shape=${bsShape.contentToString()}, elements=$bsSize")

            // Create output buffers
            val wsOutput: Any = if (wsShape.size == 2) {
                Array(wsShape[0]) { FloatArray(wsShape[1]) }
            } else {
                FloatArray(wsSize)
            }

            val bsOutput: Any = if (bsShape.size == 1) {
                FloatArray(bsSize)
            } else {
                Array(bsShape[0]) { FloatArray(bsShape[1]) }
            }

            // Run restore signature with checkpoint path
            val inputs = HashMap<String, Any>()
            inputs["checkpoint_path"] = checkpointPath
            val outputs = HashMap<String, Any>()
            outputs["ws"] = wsOutput
            outputs["bs"] = bsOutput

            interpreter.runSignature(inputs, outputs, "restore")

            println("\nAFTER 'restore' run:")
            if (wsOutput is Array<*> && wsOutput.isNotEmpty()) {
                val firstRow = wsOutput[0] as FloatArray
                println("  ws first 5 values: ${firstRow.take(5)}")
            } else if (wsOutput is FloatArray) {
                println("  ws first 5 values: ${wsOutput.take(5)}")
            }

            if (bsOutput is FloatArray) {
                println("  bs values: ${bsOutput.toList()}")
            }

            println("\n✓ DIRECT TENSOR ACCESS WORKS!")
            println("  We CAN extract weights via signature outputs!")
            println("  TRUE FedAvg IS POSSIBLE!")
            Log.i("TensorInspection", "SUCCESS: Direct tensor access confirmed!")

        } catch (e: Exception) {
            println("ERROR with 'restore' signature: ${e.message}")
            Log.e("TensorInspection", "Error reading restore", e)
            e.printStackTrace()
        }

        println("=".repeat(80))
    }

    @Test
    fun testProbeCheckpointFormat() {
        println("\n" + "=".repeat(80))
        println("CHECKPOINT FORMAT PROBE")
        println("=".repeat(80))

        val checkpointDir = context.filesDir.resolve("fedavg_test")
        checkpointDir.mkdirs()
        val checkpointPath = checkpointDir.resolve("probe.ckpt").absolutePath

        imageTrainer.saveWeights(checkpointPath)

        val file = java.io.File(checkpointPath)
        val bytes = file.readBytes()
        println("File size: ${bytes.size} bytes")
        println("Expected weight data: ${(62720 * 4 + 4) * 4} bytes")
        println("Overhead: ${bytes.size - (62720 * 4 + 4) * 4} bytes")

        // Show first 300 bytes as hex (should contain header)
        println("\nFirst 300 bytes (hex):")
        for (i in 0 until minOf(300, bytes.size)) {
            if (i % 32 == 0) print("\n  [$i]: ")
            print("%02X ".format(bytes[i]))
        }
        println()

        // Show last 300 bytes
        val lastStart = maxOf(0, bytes.size - 300)
        println("\nLast 300 bytes (hex) starting at $lastStart:")
        for (i in lastStart until bytes.size) {
            if ((i - lastStart) % 32 == 0) print("\n  [$i]: ")
            print("%02X ".format(bytes[i]))
        }
        println()

        // Find where first long zero run starts (weight data)
        var zeroRunStart = -1
        var consecutive = 0
        for (i in bytes.indices) {
            if (bytes[i] == 0.toByte()) {
                consecutive++
                if (consecutive >= 64 && zeroRunStart == -1) {
                    zeroRunStart = i - 63
                }
            } else {
                consecutive = 0
            }
        }
        println("\nFirst 64+ consecutive zero bytes start at: $zeroRunStart")

        // Find last non-zero byte
        var lastNonZero = -1
        for (i in bytes.indices.reversed()) {
            if (bytes[i] != 0.toByte()) {
                lastNonZero = i
                break
            }
        }
        println("Last non-zero byte at: $lastNonZero")

        // Count total zero and non-zero bytes
        val zeroCount = bytes.count { it == 0.toByte() }
        println("Zero bytes: $zeroCount / ${bytes.size}")
        println("Non-zero bytes: ${bytes.size - zeroCount}")

        // Show bytes around the zero-run boundary
        if (zeroRunStart > 0) {
            println("\nBytes around zero-run start ($zeroRunStart):")
            val from = maxOf(0, zeroRunStart - 16)
            val to = minOf(bytes.size, zeroRunStart + 16)
            for (i in from until to) {
                if ((i - from) % 32 == 0) print("\n  [$i]: ")
                print("%02X ".format(bytes[i]))
            }
            println()
        }

        println("\n" + "=".repeat(80))
    }

    @Test
    fun testExtractAndSetWeights() {
        println("\n" + "=".repeat(80))
        println("EXTRACT & SET WEIGHTS TEST")
        println("=".repeat(80))

        val checkpointDir = context.filesDir.resolve("fedavg_test")
        checkpointDir.mkdirs()
        val dirPath = checkpointDir.absolutePath

        // Step 1: Extract current weights
        println("\n--- Step 1: Extract weights ---")
        val original = imageTrainer.extractWeights(dirPath)
        if (original == null) {
            println("FAIL: extractWeights returned null")
            return
        }
        println("ws shape: [${original.ws.size}, ${original.ws[0].size}]")
        println("bs length: ${original.bs.size}")
        println("ws[0] first 4: ${original.ws[0].take(4)}")
        println("ws[100] first 4: ${original.ws[100].take(4)}")
        println("bs: ${original.bs.toList()}")
        Log.i("FedAvgTest", "extractWeights OK: ws=${original.ws.size}x${original.ws[0].size}, bs=${original.bs.size}")

        // Step 2: Create modified weights (add 0.5 to everything)
        println("\n--- Step 2: Create modified weights ---")
        val modifiedWs = Array(original.ws.size) { i ->
            FloatArray(original.ws[i].size) { j -> original.ws[i][j] + 0.5f }
        }
        val modifiedBs = FloatArray(original.bs.size) { i -> original.bs[i] + 0.5f }
        val modified = TransferLearningHelper.ModelWeights(ws = modifiedWs, bs = modifiedBs)
        println("Modified ws[0] first 4: ${modified.ws[0].take(4)}")
        println("Modified bs: ${modified.bs.toList()}")

        // Step 3: Set the modified weights
        println("\n--- Step 3: Set modified weights ---")
        val setResult = imageTrainer.setWeights(modified, dirPath)
        println("setWeights result: $setResult")
        if (!setResult) {
            println("FAIL: setWeights returned false")
            return
        }
        Log.i("FedAvgTest", "setWeights OK")

        // Step 4: Extract again and verify
        println("\n--- Step 4: Verify weights changed ---")
        val afterSet = imageTrainer.extractWeights(dirPath)
        if (afterSet == null) {
            println("FAIL: second extractWeights returned null")
            return
        }
        println("After set ws[0] first 4: ${afterSet.ws[0].take(4)}")
        println("After set ws[100] first 4: ${afterSet.ws[100].take(4)}")
        println("After set bs: ${afterSet.bs.toList()}")

        // Check values match what we set
        var wsMatch = true
        var wsMismatchCount = 0
        for (i in original.ws.indices) {
            for (j in original.ws[i].indices) {
                val expected = original.ws[i][j] + 0.5f
                val actual = afterSet.ws[i][j]
                if (Math.abs(expected - actual) > 0.001f) {
                    wsMatch = false
                    wsMismatchCount++
                    if (wsMismatchCount <= 5) {
                        println("ws MISMATCH at [$i][$j]: expected=$expected, actual=$actual")
                    }
                }
            }
        }

        var bsMatch = true
        for (i in original.bs.indices) {
            val expected = original.bs[i] + 0.5f
            val actual = afterSet.bs[i]
            if (Math.abs(expected - actual) > 0.001f) {
                bsMatch = false
                println("bs MISMATCH at [$i]: expected=$expected, actual=$actual")
            }
        }

        println("\n--- Results ---")
        println("ws match: $wsMatch (mismatches: $wsMismatchCount / ${original.ws.size * original.ws[0].size})")
        println("bs match: $bsMatch")

        if (wsMatch && bsMatch) {
            println("\nPASS: extract -> modify -> set -> extract roundtrip works!")
            Log.i("FedAvgTest", "PASS: Full roundtrip verified")
        } else {
            println("\nFAIL: Weight roundtrip has mismatches")
            Log.e("FedAvgTest", "FAIL: ws=$wsMatch, bs=$bsMatch, wsMismatch=$wsMismatchCount")
        }

        println("=".repeat(80))
    }

    @Test
    fun testResetCheckpoint() {
        println("\n" + "=".repeat(80))
        println("RESET CHECKPOINT TEST")
        println("=".repeat(80))

        val checkpointDir = context.filesDir.resolve("fedavg_test")
        checkpointDir.mkdirs()
        val dirPath = checkpointDir.absolutePath

        // Step 1: Extract current weights
        println("\n--- Step 1: Extract current weights ---")
        val before = imageTrainer.extractWeights(dirPath)
        if (before == null) {
            println("FAIL: extractWeights returned null")
            return
        }
        println("Before reset ws[0] first 4: ${before.ws[0].take(4)}")
        println("Before reset bs: ${before.bs.toList()}")

        // Step 2: Modify weights (so we can verify reset changes them)
        println("\n--- Step 2: Set modified weights ---")
        val modifiedWs = Array(before.ws.size) { i ->
            FloatArray(before.ws[i].size) { j -> 1.0f } // Set all to 1.0
        }
        val modifiedBs = FloatArray(before.bs.size) { 1.0f }
        val modified = TransferLearningHelper.ModelWeights(ws = modifiedWs, bs = modifiedBs)
        imageTrainer.setWeights(modified, dirPath)
        
        val afterModify = imageTrainer.extractWeights(dirPath)!!
        println("After modify ws[0] first 4: ${afterModify.ws[0].take(4)}")
        println("After modify bs: ${afterModify.bs.toList()}")

        // Step 3: Reset checkpoint
        println("\n--- Step 3: Reset checkpoint ---")
        val resetResult = imageTrainer.resetCheckpoint(dirPath)
        println("resetCheckpoint result: $resetResult")
        if (!resetResult) {
            println("FAIL: resetCheckpoint returned false")
            return
        }

        // Step 4: Extract weights after reset
        println("\n--- Step 4: Verify weights changed after reset ---")
        val afterReset = imageTrainer.extractWeights(dirPath)
        if (afterReset == null) {
            println("FAIL: extractWeights after reset returned null")
            return
        }
        println("After reset ws[0] first 4: ${afterReset.ws[0].take(4)}")
        println("After reset bs: ${afterReset.bs.toList()}")

        // Check weights are different from the modified values (all 1.0)
        val wsChanged = afterReset.ws[0][0] != 1.0f || afterReset.ws[0][1] != 1.0f
        val bsChanged = afterReset.bs[0] != 1.0f || afterReset.bs[1] != 1.0f

        println("\n--- Results ---")
        println("ws changed from 1.0: $wsChanged")
        println("bs changed from 1.0: $bsChanged")

        if (wsChanged && bsChanged) {
            println("\nPASS: Reset produced new random weights!")
            Log.i("ResetTest", "PASS: Reset verified")
        } else {
            println("\nFAIL: Reset did not change weights")
            Log.e("ResetTest", "FAIL: weights still 1.0 after reset")
        }

        println("=".repeat(80))
    }

    @Test
    fun testFederatedAveraging() {
        println("\n" + "=".repeat(80))
        println("FEDERATED AVERAGING TEST")
        println("=".repeat(80))

        val checkpointDir = context.filesDir.resolve("fedavg_test")
        checkpointDir.mkdirs()
        val dirPath = checkpointDir.absolutePath

        // Create mock weights from 3 "clients"
        println("\n--- Creating mock client weights ---")
        
        // Client 1: weights are all 0.0
        val client1Ws = Array(62720) { FloatArray(4) { 0.0f } }
        val client1Bs = FloatArray(4) { 0.0f }
        val client1 = TransferLearningHelper.ModelWeights(ws = client1Ws, bs = client1Bs)
        
        // Client 2: weights are all 1.0
        val client2Ws = Array(62720) { FloatArray(4) { 1.0f } }
        val client2Bs = FloatArray(4) { 1.0f }
        val client2 = TransferLearningHelper.ModelWeights(ws = client2Ws, bs = client2Bs)
        
        // Client 3: weights are all 0.5
        val client3Ws = Array(62720) { FloatArray(4) { 0.5f } }
        val client3Bs = FloatArray(4) { 0.5f }
        val client3 = TransferLearningHelper.ModelWeights(ws = client3Ws, bs = client3Bs)

        println("Client 1 ws[0]: ${client1.ws[0].toList()}, bs: ${client1.bs.toList()}")
        println("Client 2 ws[0]: ${client2.ws[0].toList()}, bs: ${client2.bs.toList()}")
        println("Client 3 ws[0]: ${client3.ws[0].toList()}, bs: ${client3.bs.toList()}")

        // Test 1: Equal sample counts (should be simple average)
        println("\n--- Test 1: Equal sample counts (100, 100, 100) ---")
        val weightsList = listOf(client1, client2, client3)
        val equalCounts = listOf(100, 100, 100)
        
        val avgEqual = imageTrainer.performFederatedAveraging(weightsList, equalCounts)
        if (avgEqual == null) {
            println("FAIL: performFederatedAveraging returned null")
            return
        }
        
        // Expected: (0.0 + 1.0 + 0.5) / 3 = 0.5
        val expectedEqual = 0.5f
        println("Result ws[0][0]: ${avgEqual.ws[0][0]} (expected: $expectedEqual)")
        println("Result bs[0]: ${avgEqual.bs[0]} (expected: $expectedEqual)")
        
        val equalWsOk = Math.abs(avgEqual.ws[0][0] - expectedEqual) < 0.001f
        val equalBsOk = Math.abs(avgEqual.bs[0] - expectedEqual) < 0.001f
        println("Equal counts test: ws=${if(equalWsOk) "PASS" else "FAIL"}, bs=${if(equalBsOk) "PASS" else "FAIL"}")

        // Test 2: Weighted sample counts (100, 200, 100) - client2 has double weight
        println("\n--- Test 2: Weighted sample counts (100, 200, 100) ---")
        val weightedCounts = listOf(100, 200, 100)
        
        val avgWeighted = imageTrainer.performFederatedAveraging(weightsList, weightedCounts)
        if (avgWeighted == null) {
            println("FAIL: performFederatedAveraging with weighted counts returned null")
            return
        }
        
        // Expected: (0.0*100 + 1.0*200 + 0.5*100) / 400 = (0 + 200 + 50) / 400 = 0.625
        val expectedWeighted = 0.625f
        println("Result ws[0][0]: ${avgWeighted.ws[0][0]} (expected: $expectedWeighted)")
        println("Result bs[0]: ${avgWeighted.bs[0]} (expected: $expectedWeighted)")
        
        val weightedWsOk = Math.abs(avgWeighted.ws[0][0] - expectedWeighted) < 0.001f
        val weightedBsOk = Math.abs(avgWeighted.bs[0] - expectedWeighted) < 0.001f
        println("Weighted counts test: ws=${if(weightedWsOk) "PASS" else "FAIL"}, bs=${if(weightedBsOk) "PASS" else "FAIL"}")

        // Test 3: Apply averaged weights and verify
        println("\n--- Test 3: Apply averaged weights to model ---")
        val applyResult = imageTrainer.setWeights(avgWeighted, dirPath)
        println("setWeights result: $applyResult")
        
        val afterApply = imageTrainer.extractWeights(dirPath)
        if (afterApply == null) {
            println("FAIL: extractWeights after apply returned null")
            return
        }
        println("After apply ws[0][0]: ${afterApply.ws[0][0]}")
        println("After apply bs[0]: ${afterApply.bs[0]}")
        
        val applyWsOk = Math.abs(afterApply.ws[0][0] - expectedWeighted) < 0.001f
        val applyBsOk = Math.abs(afterApply.bs[0] - expectedWeighted) < 0.001f
        println("Apply test: ws=${if(applyWsOk) "PASS" else "FAIL"}, bs=${if(applyBsOk) "PASS" else "FAIL"}")

        // Summary
        println("\n--- Summary ---")
        val allPass = equalWsOk && equalBsOk && weightedWsOk && weightedBsOk && applyWsOk && applyBsOk
        if (allPass) {
            println("PASS: All FedAvg tests passed!")
            Log.i("FedAvgTest", "PASS: FedAvg algorithm verified")
        } else {
            println("FAIL: Some tests failed")
            Log.e("FedAvgTest", "FAIL: FedAvg has issues")
        }

        println("=".repeat(80))
    }

    @Test
    fun testPayloadSerialization() {
        println("\n" + "=".repeat(80))
        println("PAYLOAD SERIALIZATION TEST (sample count + weights)")
        println("=".repeat(80))

        val checkpointDir = context.filesDir.resolve("fedavg_test")
        checkpointDir.mkdirs()

        // save a checkpoint to get real bytes
        val ckptPath = checkpointDir.resolve("serial_test.ckpt").absolutePath
        imageTrainer.saveWeights(ckptPath)
        val ckptBytes = java.io.File(ckptPath).readBytes()
        val sampleCount = 42

        // encode: 4 bytes sample count + checkpoint bytes
        println("\n--- Encoding ---")
        val buf = java.nio.ByteBuffer.allocate(4 + ckptBytes.size)
        buf.order(java.nio.ByteOrder.BIG_ENDIAN)
        buf.putInt(sampleCount)
        buf.put(ckptBytes)
        val payload = buf.array()
        println("Checkpoint: ${ckptBytes.size} bytes")
        println("Payload: ${payload.size} bytes (4 header + ${ckptBytes.size} body)")
        println("First 8 bytes: ${payload.take(8).map { "%02X".format(it) }}")

        // decode: parse sample count + checkpoint
        println("\n--- Decoding ---")
        val readBuf = java.nio.ByteBuffer.wrap(payload, 0, 4)
        readBuf.order(java.nio.ByteOrder.BIG_ENDIAN)
        val parsedCount = readBuf.getInt()
        val parsedCkpt = payload.copyOfRange(4, payload.size)

        println("Parsed sample count: $parsedCount (expected: $sampleCount)")
        println("Parsed checkpoint size: ${parsedCkpt.size} (expected: ${ckptBytes.size})")

        val countOk = parsedCount == sampleCount
        val sizeOk = parsedCkpt.size == ckptBytes.size
        val bytesOk = parsedCkpt.contentEquals(ckptBytes)

        println("\n--- Results ---")
        println("Sample count match: $countOk")
        println("Size match: $sizeOk")
        println("Bytes match: $bytesOk")

        // verify the parsed checkpoint can still be loaded
        val parsedFile = checkpointDir.resolve("parsed_test.ckpt")
        parsedFile.writeBytes(parsedCkpt)
        val restoreOk = imageTrainer.restoreWeights(parsedFile.absolutePath)
        println("Restore from parsed checkpoint: $restoreOk")

        val allPass = countOk && sizeOk && bytesOk && restoreOk
        if (allPass) {
            println("\nPASS: Payload serialization roundtrip works!")
            Log.i("PayloadTest", "PASS")
        } else {
            println("\nFAIL: Serialization has issues")
            Log.e("PayloadTest", "FAIL")
        }

        println("=".repeat(80))
    }

    @Test
    fun testWeightedFedAvgWithRealCounts() {
        println("\n" + "=".repeat(80))
        println("WEIGHTED FEDAVG WITH REAL SAMPLE COUNTS")
        println("=".repeat(80))

        // client A: all 2.0, trained on 30 samples
        val aWs = Array(62720) { FloatArray(4) { 2.0f } }
        val aBs = FloatArray(4) { 2.0f }
        val clientA = TransferLearningHelper.ModelWeights(ws = aWs, bs = aBs)
        val samplesA = 30

        // client B: all 4.0, trained on 10 samples
        val bWs = Array(62720) { FloatArray(4) { 4.0f } }
        val bBs = FloatArray(4) { 4.0f }
        val clientB = TransferLearningHelper.ModelWeights(ws = bWs, bs = bBs)
        val samplesB = 10

        println("Client A: all 2.0, $samplesA samples")
        println("Client B: all 4.0, $samplesB samples")

        // expected: (2.0*30 + 4.0*10) / 40 = (60+40)/40 = 2.5
        val expected = 2.5f
        println("Expected avg: $expected  (formula: (2.0*30 + 4.0*10) / 40)")

        val result = imageTrainer.performFederatedAveraging(
            listOf(clientA, clientB),
            listOf(samplesA, samplesB)
        )

        if (result == null) {
            println("FAIL: FedAvg returned null")
            return
        }

        println("Result ws[0][0]: ${result.ws[0][0]}")
        println("Result bs[0]: ${result.bs[0]}")

        val wsOk = Math.abs(result.ws[0][0] - expected) < 0.001f
        val bsOk = Math.abs(result.bs[0] - expected) < 0.001f

        // also check a random middle value
        val midOk = Math.abs(result.ws[30000][2] - expected) < 0.001f
        println("Result ws[30000][2]: ${result.ws[30000][2]}")

        val allPass = wsOk && bsOk && midOk
        if (allPass) {
            println("\nPASS: Weighted FedAvg with real sample counts works!")
            Log.i("WeightedFedAvg", "PASS")
        } else {
            println("\nFAIL: Weighted FedAvg incorrect")
            Log.e("WeightedFedAvg", "FAIL: ws=$wsOk bs=$bsOk mid=$midOk")
        }

        println("=".repeat(80))
    }
}
