package com.fyp.collaborite.tests

import android.content.Context
import android.util.Log
import org.tensorflow.lite.examples.modelpersonalization.TransferLearningHelper
import java.io.File

class Phase2WeightTransmissionTest(private val context: Context) {

    private val tag = "Phase2Test"

    private val checkpointDir: File by lazy {
        File(context.filesDir, "federated_checkpoints").apply {
            if (!exists()) {
                mkdirs()
            }
        }
    }

    private val transferLearningHelper: TransferLearningHelper by lazy {
        TransferLearningHelper(
            context = context,
            classifierListener = null
        )
    }

    fun mockWeightTransfer(): Boolean {
        Log.d(tag, "=== MOCK TEST: Starting weight transfer simulation ===")

        try {
            // Step 1: Save local weights
            Log.d(tag, "Step 1: Saving local model weights...")
            val localCheckpoint = File(checkpointDir, "local_weights.ckpt")
            val saveSuccess = transferLearningHelper.saveWeights(localCheckpoint.absolutePath)

            if (!saveSuccess || !localCheckpoint.exists()) {
                Log.e(tag, "FAILED: Could not save local weights")
                return false
            }

            val localBytes = localCheckpoint.readBytes()
            Log.d(tag, "SUCCESS: Saved ${localBytes.size} bytes to ${localCheckpoint.absolutePath}")

            // Step 2: Simulate receiving from peer (copy file to peer location)
            Log.d(tag, "Step 2: Simulating peer transmission...")
            val mockPeerId = "ei:MOCK_PEER_123"
            val peerCheckpoint = File(checkpointDir, "peer_${mockPeerId}_weights.ckpt")

            peerCheckpoint.writeBytes(localBytes)
            Log.d(tag, "SUCCESS: Received ${localBytes.size} bytes from $mockPeerId")
            Log.d(tag, "         Saved to ${peerCheckpoint.absolutePath}")

            // Step 3: Verify both files exist
            Log.d(tag, "Step 3: Verifying checkpoint files...")
            val localExists = localCheckpoint.exists()
            val peerExists = peerCheckpoint.exists()

            Log.d(tag, "Local checkpoint exists: $localExists (${localCheckpoint.length()} bytes)")
            Log.d(tag, "Peer checkpoint exists: $peerExists (${peerCheckpoint.length()} bytes)")

            if (localExists && peerExists) {
                Log.d(tag, "=== MOCK TEST PASSED: Weight transfer simulation complete ===")
                return true
            } else {
                Log.e(tag, "=== MOCK TEST FAILED: File verification failed ===")
                return false
            }

        } catch (e: Exception) {
            Log.e(tag, "=== MOCK TEST ERROR: ${e.message} ===", e)
            return false
        }
    }
}
