package com.fyp.collaborite.tests

import android.content.Context
import android.util.Log
import org.tensorflow.lite.examples.modelpersonalization.TransferLearningHelper
import java.io.File

class Phase1WeightManagementTest(private val context: Context) {

    private val tag = "Phase1Test"

    fun testSaveRestoreSignatures(): Boolean {
        val helper = TransferLearningHelper(
            context = context,
            classifierListener = null
        )

        val checkpointDir = File(context.filesDir, "test_checkpoints")
        if (!checkpointDir.exists()) {
            checkpointDir.mkdirs()
        }
        val checkpointPath = File(checkpointDir, "test_weights.ckpt").absolutePath

        try {
            Log.d(tag, "Testing save signature with path: $checkpointPath")
            val saveResult = helper.saveWeights(checkpointPath)
            Log.d(tag, "Save result: $saveResult")

            val checkpointFile = File(checkpointPath)
            if (checkpointFile.exists()) {
                Log.d(tag, "Checkpoint file created: ${checkpointFile.length()} bytes")
            } else {
                Log.e(tag, "Checkpoint file NOT created")
                return false
            }

            Log.d(tag, "Testing restore signature...")
            val restoreResult = helper.restoreWeights(checkpointPath)
            Log.d(tag, "Restore result: $restoreResult")

            helper.close()
            return true

        } catch (e: Exception) {
            Log.e(tag, "Error testing weight management: ${e.message}", e)
            return false
        }
    }

    fun testWeightExtraction(): ByteArray? {
        val helper = TransferLearningHelper(
            context = context,
            classifierListener = null
        )

        try {
            val checkpointDir = File(context.filesDir, "temp_weights")
            if (!checkpointDir.exists()) {
                checkpointDir.mkdirs()
            }
            val checkpointPath = File(checkpointDir, "weights.ckpt").absolutePath

            helper.saveWeights(checkpointPath)

            val checkpointFile = File(checkpointPath)
            if (checkpointFile.exists()) {
                val weightBytes = checkpointFile.readBytes()
                Log.d(tag, "Extracted ${weightBytes.size} bytes of weight data")
                helper.close()
                return weightBytes
            }

            helper.close()
            return null

        } catch (e: Exception) {
            Log.e(tag, "Error extracting weights: ${e.message}", e)
            return null
        }
    }
}
