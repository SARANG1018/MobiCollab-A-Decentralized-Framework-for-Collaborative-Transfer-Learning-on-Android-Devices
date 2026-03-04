package com.fyp.collaborite.tests

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.fyp.collaborite.distributed.wifi.WifiKtsManager
import org.tensorflow.lite.examples.modelpersonalization.TransferLearningHelper
import java.io.File

class Phase4TrainingIntegrationTest(private val context: Context) {

    private val tag = "Phase4Test"

    // Test the complete training workflow
    fun testTrainingWorkflow(): Boolean {
        Log.d(tag, "=== PHASE 4 TEST: Training Integration ===")

        val helper = TransferLearningHelper(
            context = context,
            classifierListener = null
        )

        val checkpointDir = File(context.filesDir, "federated_checkpoints")
        if (!checkpointDir.exists()) {
            checkpointDir.mkdirs()
        }

        try {
            // Step 1: Add minimum samples
            Log.d(tag, "Step 1: Adding 20 training samples...")
            val dummyBitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)

            for (i in 1..10) {
                helper.addSample(dummyBitmap, "1", 0)
            }
            for (i in 1..10) {
                helper.addSample(dummyBitmap, "2", 0)
            }

            val sampleCount = helper.getSampleCount()
            Log.d(tag, "Added $sampleCount samples")

            if (sampleCount < 20) {
                Log.e(tag, "FAILED: Not enough samples added")
                return false
            }

            // Step 2: Start training
            Log.d(tag, "Step 2: Starting training...")
            helper.startTraining()
            Log.d(tag, "Training started")

            // Wait a bit for training to run
            Thread.sleep(3000)

            // Step 3: Stop training
            Log.d(tag, "Step 3: Pausing training...")
            helper.pauseTraining()
            Log.d(tag, "Training paused")

            // Step 4: Save weights
            Log.d(tag, "Step 4: Saving trained weights...")
            val checkpointPath = File(checkpointDir, "trained_weights.ckpt").absolutePath
            val saveSuccess = helper.saveWeights(checkpointPath)

            if (!saveSuccess) {
                Log.e(tag, "FAILED: Could not save weights")
                return false
            }

            val checkpointFile = File(checkpointPath)
            if (!checkpointFile.exists()) {
                Log.e(tag, "FAILED: Checkpoint file not created")
                return false
            }

            Log.d(tag, "SUCCESS: Checkpoint saved (${checkpointFile.length()} bytes)")

            // Step 5: Verify we can restore weights
            Log.d(tag, "Step 5: Testing weight restoration...")
            val restoreSuccess = helper.restoreWeights(checkpointPath)

            if (!restoreSuccess) {
                Log.e(tag, "FAILED: Could not restore weights")
                return false
            }

            Log.d(tag, "SUCCESS: Weights restored")

            helper.close()

            Log.d(tag, "=== PHASE 4 TEST PASSED ===")
            return true

        } catch (e: Exception) {
            Log.e(tag, "=== PHASE 4 TEST ERROR: ${e.message} ===", e)
            return false
        }
    }
}
