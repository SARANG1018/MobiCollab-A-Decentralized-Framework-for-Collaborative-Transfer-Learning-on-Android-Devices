package com.fyp.collaborite.distributed.wifi

import android.util.Log
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import com.fyp.collaborite.ConnectionActivity
import com.google.android.gms.nearby.Nearby
import com.google.android.gms.nearby.connection.AdvertisingOptions
import com.google.android.gms.nearby.connection.ConnectionInfo
import com.google.android.gms.nearby.connection.ConnectionLifecycleCallback
import com.google.android.gms.nearby.connection.ConnectionResolution
import com.google.android.gms.nearby.connection.ConnectionsClient
import com.google.android.gms.nearby.connection.DiscoveredEndpointInfo
import com.google.android.gms.nearby.connection.DiscoveryOptions
import com.google.android.gms.nearby.connection.EndpointDiscoveryCallback
import com.google.android.gms.nearby.connection.Payload
import com.google.android.gms.nearby.connection.PayloadCallback
import com.google.android.gms.nearby.connection.PayloadTransferUpdate
import com.google.android.gms.nearby.connection.Strategy
import org.tensorflow.lite.examples.modelpersonalization.TransferLearningHelper
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.delay


class WifiKtsManager(val activity: ConnectionActivity) {
    lateinit var myCodeName:String
    val packageName="com.fyp.collaborite"
    val peers= mutableStateListOf<String>()
    val sampleCount= mutableStateOf<Int>(0)

    val finalVal= mutableStateOf(0)
    val connectedPeers= mutableStateListOf<String>()
    var peerMap= mutableStateMapOf<String,DiscoveredEndpointInfo>()

    // training state
    private val MIN_SAMPLES_FOR_TRAINING = 20
    private var isTraining = false

    // federated learning sync state
    val syncStatus = mutableStateOf("Not synced")
    private val receivedPeerWeights = mutableMapOf<String, File>()
    private val peerSampleCounts = mutableMapOf<String, Int>()
    private var isSyncing = false
    private var waitingForPeers = false

    private val checkpointDir: File by lazy {
        File(activity.filesDir, "federated_checkpoints").apply {
            if (!exists()) {
                mkdirs()
                Log.d("WIFI", "Created checkpoint directory: $absolutePath")
            }
        }
    }

    public lateinit var transferLearningHelper: TransferLearningHelper;
    init {
        transferLearningHelper = TransferLearningHelper(
            context=activity,
            classifierListener = activity
        )
    }

    // Manually start training (triggered by Train button)
    fun startTrainingManually() {
        val currentSamples = transferLearningHelper.getSampleCount()

        if (currentSamples < MIN_SAMPLES_FOR_TRAINING) {
            Log.e("WIFI", "Not enough samples to train: $currentSamples/$MIN_SAMPLES_FOR_TRAINING")
            android.widget.Toast.makeText(
                activity,
                "Need at least $MIN_SAMPLES_FOR_TRAINING samples (currently: $currentSamples)",
                android.widget.Toast.LENGTH_SHORT
            ).show()
            return
        }

        if (isTraining) {
            Log.w("WIFI", "Training already in progress")
            android.widget.Toast.makeText(activity, "Training already running", android.widget.Toast.LENGTH_SHORT).show()
            return
        }

        Log.d("WIFI", "Starting training with $currentSamples samples...")
        isTraining = true

        try {
            transferLearningHelper.startTraining()
            Log.d("WIFI", "Training started successfully")
            android.widget.Toast.makeText(activity, "Training started!", android.widget.Toast.LENGTH_SHORT).show()

        } catch (e: Exception) {
            Log.e("WIFI", "Failed to start training: ${e.message}", e)
            isTraining = false
            android.widget.Toast.makeText(activity, "Training failed: ${e.message}", android.widget.Toast.LENGTH_LONG).show()
        }
    }

    // stop training, send weights, wait for peers, run fedavg, inject result
    fun syncWeightsWithPeers() {
        if (isSyncing) {
            android.widget.Toast.makeText(activity, "Sync already in progress", android.widget.Toast.LENGTH_SHORT).show()
            return
        }

        // pause training if running
        if (isTraining) {
            transferLearningHelper.pauseTraining()
            isTraining = false
        }

        isSyncing = true
        syncStatus.value = "Sending weights..."
        receivedPeerWeights.clear()
        peerSampleCounts.clear()

        // send weights to all peers
        val sent = sendModelWeights()
        if (!sent && connectedPeers.isNotEmpty()) {
            syncStatus.value = "Send failed"
            isSyncing = false
            return
        }

        if (connectedPeers.isEmpty()) {
            // no peers - just keep local weights
            syncStatus.value = "No peers connected"
            isSyncing = false
            android.widget.Toast.makeText(activity, "No peers to sync with", android.widget.Toast.LENGTH_SHORT).show()
            return
        }

        // wait for peer weights to arrive
        waitingForPeers = true
        syncStatus.value = "Waiting for ${connectedPeers.size} peer(s)..."

        CoroutineScope(Dispatchers.IO).launch {
            val timeout = 60_000L // 60 seconds
            val startTime = System.currentTimeMillis()

            while (receivedPeerWeights.size < connectedPeers.size) {
                if (System.currentTimeMillis() - startTime > timeout) {
                    Log.w("WIFI", "Timeout waiting for peers. Got ${receivedPeerWeights.size}/${connectedPeers.size}")
                    break
                }
                delay(500)
            }

            waitingForPeers = false
            performFedAvgAndInject()
        }
    }

    // run federated averaging with local + received peer weights
    private fun performFedAvgAndInject() {
        try {
            syncStatus.value = "Running FedAvg..."

            // extract local weights
            val localWeights = transferLearningHelper.extractWeights(checkpointDir.absolutePath)
            if (localWeights == null) {
                Log.e("WIFI", "Failed to extract local weights")
                syncStatus.value = "FedAvg failed: local extract error"
                isSyncing = false
                return
            }

            val allWeights = mutableListOf(localWeights)
            val allSampleCounts = mutableListOf(transferLearningHelper.getSampleCount())

            // extract each peer's weights from their checkpoint files
            for ((peerId, peerFile) in receivedPeerWeights) {
                try {
                    // restore peer checkpoint into model temporarily
                    val restoreOk = transferLearningHelper.restoreWeights(peerFile.absolutePath)
                    if (!restoreOk) {
                        Log.e("WIFI", "Failed to restore peer $peerId checkpoint")
                        continue
                    }

                    val peerWeights = transferLearningHelper.extractWeights(checkpointDir.absolutePath)
                    if (peerWeights != null) {
                        allWeights.add(peerWeights)
                        allSampleCounts.add(peerSampleCounts[peerId] ?: transferLearningHelper.getSampleCount())
                    }
                } catch (e: Exception) {
                    Log.e("WIFI", "Error extracting peer $peerId weights: ${e.message}")
                }
            }

            Log.d("WIFI", "FedAvg: ${allWeights.size} participants")

            if (allWeights.size < 2) {
                // restore our own weights back since we may have overwritten them
                val localCkpt = File(checkpointDir, "local_weights.ckpt").absolutePath
                transferLearningHelper.restoreWeights(localCkpt)
                syncStatus.value = "FedAvg skipped: not enough peers"
                isSyncing = false
                return
            }

            // run federated averaging
            val averaged = transferLearningHelper.performFederatedAveraging(allWeights, allSampleCounts)
            if (averaged == null) {
                Log.e("WIFI", "FedAvg returned null")
                val localCkpt = File(checkpointDir, "local_weights.ckpt").absolutePath
                transferLearningHelper.restoreWeights(localCkpt)
                syncStatus.value = "FedAvg failed"
                isSyncing = false
                return
            }

            // inject averaged weights into the model
            val injected = transferLearningHelper.setWeights(averaged, checkpointDir.absolutePath)
            if (!injected) {
                Log.e("WIFI", "Failed to inject averaged weights")
                syncStatus.value = "FedAvg inject failed"
                isSyncing = false
                return
            }

            syncStatus.value = "Synced ✓ (${allWeights.size} devices)"
            isSyncing = false

            CoroutineScope(Dispatchers.Main).launch {
                android.widget.Toast.makeText(
                    activity,
                    "FedAvg done! ${allWeights.size} devices averaged",
                    android.widget.Toast.LENGTH_SHORT
                ).show()
            }

            Log.d("WIFI", "FedAvg complete. ${allWeights.size} devices averaged.")

        } catch (e: Exception) {
            Log.e("WIFI", "FedAvg error: ${e.message}", e)
            syncStatus.value = "FedAvg error"
            isSyncing = false
        }
    }

    fun sendModelWeights(): Boolean {
        if (connectedPeers.isEmpty()) {
            Log.d("WIFI", "No connected peers to send weights to")
            return false
        }

        try {
            // Step 1: Save current model weights to checkpoint file
            val checkpointPath = File(checkpointDir, "local_weights.ckpt").absolutePath
            val saveSuccess = transferLearningHelper.saveWeights(checkpointPath)

            if (!saveSuccess) {
                Log.e("WIFI", "Failed to save weights to checkpoint")
                return false
            }

            // Step 2: Read checkpoint file as byte array
            val checkpointFile = File(checkpointPath)
            if (!checkpointFile.exists()) {
                Log.e("WIFI", "Checkpoint file not found after save")
                return false
            }

            val weightBytes = checkpointFile.readBytes()
            val sampleCount = transferLearningHelper.getSampleCount()

            // prepend 4 bytes of sample count to weight bytes
            val buf = java.nio.ByteBuffer.allocate(4 + weightBytes.size)
            buf.order(java.nio.ByteOrder.BIG_ENDIAN)
            buf.putInt(sampleCount)
            buf.put(weightBytes)
            val payload = buf.array()

            Log.d("WIFI", "Sending ${payload.size} bytes (4 header + ${weightBytes.size} weights, $sampleCount samples)")

            connectionsClient.sendPayload(
                connectedPeers,
                Payload.fromBytes(payload)
            ).addOnCompleteListener { task ->
                if (task.isSuccessful) {
                    Log.d("WIFI", "Model weights sent successfully to ${connectedPeers.size} peer(s)")
                } else if (task.exception != null) {
                    Log.e("WIFI", "Failed to send weights: ${task.exception!!.message}")
                }
            }

            return true

        } catch (e: Exception) {
            Log.e("WIFI", "Error sending model weights: ${e.message}", e)
            return false
        }
    }

    private val STRATEGY = Strategy.P2P_STAR
    val endpointDiscoveryCallback = object : EndpointDiscoveryCallback() {
        override fun onEndpointFound(endpointId: String, info: DiscoveredEndpointInfo) {
            Log.d("WIFI","Client Found")
            Log.d("WIFI",endpointId)
            Log.d("WIFI",info.toString())
            peers.add(endpointId)


            peerMap.put(endpointId,info)


        }

        override fun onEndpointLost(endpointId: String) {
            peers.remove(endpointId);
            connectedPeers.remove(endpointId);
            peerMap.remove(endpointId);
        }
    }
    fun xstartAdvertising(codeName: String) {
        myCodeName = "TA$codeName"
        val options = AdvertisingOptions.Builder().setStrategy(STRATEGY).build()
        connectionsClient.startAdvertising(
            myCodeName,
            packageName,
            connectionLifecycleCallback,
            options
        ).addOnCompleteListener {
            if (it.isSuccessful) {
                Log.d("WIFI", "KT Advertising")
            } else if (it.exception != null) {
                Log.d("WIFI", "Exception Occured Advertising:${it.exception!!.message}")
            }
        }
    }

    fun connectPeer(endpointId: String) {
        connectionsClient.requestConnection(myCodeName, endpointId, connectionLifecycleCallback).addOnCompleteListener {
            if(it.isSuccessful){
                Log.d("WIFI","Requested Connection")
                connectedPeers.add(endpointId)
            }else if(it.exception!=null){
                Log.d("WIFI","Exception Occured Requesting:${it.exception!!.message}")
            }

        }
    }
    fun xstartDiscovery() {
        val options = DiscoveryOptions.Builder().setStrategy(STRATEGY).build()
        connectionsClient.startDiscovery(packageName, endpointDiscoveryCallback, options).addOnCompleteListener {
            if (it.isSuccessful) {
                Log.d("WIFI", "KT Discovery Successful")
            } else if (it.exception != null) {
                Log.d("WIFI", "Exception Occured Discovery:${it.exception!!.message}")
            }
        }
    }

    val connectionLifecycleCallback = object : ConnectionLifecycleCallback() {
        override fun onConnectionInitiated(endpointId: String, info: ConnectionInfo) {
            Log.d("WIFI", "Connection Accepted ${info.endpointName}")
            connectionsClient.acceptConnection(endpointId, payloadCallback)
        }
        private val payloadCallback: PayloadCallback = object : PayloadCallback() {
            override fun onPayloadReceived(endpointId: String, payload: Payload) {
                Log.d("WIFI","PAYLOAD RECEIVED from $endpointId");

                // OLD: Image reception (replaced below)
                /*
                payload.asBytes()?.let {
                   Log.d("WIFI",String(it, UTF_8));
//                    weightInitialized.put(endpointId,Weights.fromParcableString(String(it, UTF_8)))
                    val bmp = BitmapFactory.decodeByteArray(it, 0, it.size)
                    //TODO Update with real values
                    transferLearningHelper.addSample(bmp,"1",0);
                    sampleCount.value=transferLearningHelper.getSampleCount()
                }
//                transferLearningHelper.startTraining();
//
//                Timer("Stopping Training", false).schedule(1500) {
//                    transferLearningHelper.pauseTraining()
//                }

//                finalVal.value=Weights.calculateWeights(weightInitialized.values.plus(currentWeight.value) as List<Weights>)
                */

                // parse sample count (first 4 bytes) + checkpoint bytes
                payload.asBytes()?.let { rawBytes ->
                    try {
                        if (rawBytes.size < 5) {
                            Log.e("WIFI", "Payload too small: ${rawBytes.size} bytes")
                            return@let
                        }

                        val buf = java.nio.ByteBuffer.wrap(rawBytes, 0, 4)
                        buf.order(java.nio.ByteOrder.BIG_ENDIAN)
                        val peerSamples = buf.getInt()
                        val ckptBytes = rawBytes.copyOfRange(4, rawBytes.size)

                        val peerCheckpointFile = File(checkpointDir, "peer_${endpointId}_weights.ckpt")
                        peerCheckpointFile.writeBytes(ckptBytes)

                        Log.d("WIFI", "Peer $endpointId: $peerSamples samples, ${ckptBytes.size} weight bytes")

                        receivedPeerWeights[endpointId] = peerCheckpointFile
                        peerSampleCounts[endpointId] = peerSamples

                        if (!isSyncing && !waitingForPeers) {
                            Log.d("WIFI", "Received peer weights outside sync. Stored for next sync.")
                        }
                        Unit
                    } catch (e: Exception) {
                        Log.e("WIFI", "Failed to save peer weights: ${e.message}", e)
                    }
                }
            }

            override fun onPayloadTransferUpdate(endpointId: String, update: PayloadTransferUpdate) {
            }
        }

        override fun onConnectionResult(endpointId: String, result: ConnectionResolution) {
            if (result.status.isSuccess) {
                connectedPeers.add(endpointId);
                connectionsClient.stopAdvertising()
                connectionsClient.stopDiscovery()
//                opponentEndpointId = endpointId
//                binding.opponentName.text = opponentName
//                binding.status.text = "Connected"
//                setGameControllerEnabled(true) // we can start playing
            }
        }

        override fun onDisconnected(endpointId: String) {
            resetGame()
        }
    }

    private fun resetGame() {



    private lateinit var connectionsClient: ConnectionsClient
    init{
        connectionsClient = Nearby.getConnectionsClient(activity)
    }
}