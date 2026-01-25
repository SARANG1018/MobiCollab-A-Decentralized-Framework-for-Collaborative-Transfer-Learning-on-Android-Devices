package com.fyp.collaborite.distributed.wifi

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.HandlerThread
import android.os.Looper
import android.util.Log
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.MutableLiveData
import com.fyp.collaborite.ConnectionActivity
import com.fyp.collaborite.WifiActivity
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
import java.io.ByteArrayOutputStream
import java.util.Timer
import java.util.concurrent.Future
import java.util.logging.Handler
import kotlin.concurrent.schedule
import kotlin.text.Charsets.UTF_8


class WifiKtsManager(val activity: ConnectionActivity) {
    lateinit var myCodeName:String
    val packageName="com.fyp.collaborite"
    val peers= mutableStateListOf<String>()
    val sampleCount= mutableStateOf<Int>(0)

    val finalVal= mutableStateOf(0)
    val connectedPeers= mutableStateListOf<String>()
    var peerMap= mutableStateMapOf<String,DiscoveredEndpointInfo>()

    public lateinit var transferLearningHelper: TransferLearningHelper;
    init {
        transferLearningHelper = TransferLearningHelper(
            context = activity.applicationContext,
            classifierListener = activity
        )
    }


    private fun bitmapToByte(image: Bitmap): ByteArray {
        val stream = ByteArrayOutputStream()
        image.compress(Bitmap.CompressFormat.PNG, 90, stream)

        return stream.toByteArray()
    }
    fun sendWeights(bitmap: Bitmap,className: String) {
        if(className!="1"){
            Log.d("WIFI","SEND ONLY True Images")
            return;
        }



        connectionsClient.sendPayload(
            connectedPeers,
            Payload.fromBytes(bitmapToByte(bitmap))
        ).addOnCompleteListener {
            if(it.isSuccessful){
                Log.d("WIFI","Payload Sending Success")
            }else if(it.exception!=null){
                Log.d("WIFI","Exception Occured Sending Data:${it.exception!!.message}")
            }
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
     fun xstartAdvertising(codeName:String) {
         myCodeName="TA$codeName"
        val options = AdvertisingOptions.Builder().setStrategy(STRATEGY).build()
        // Note: Advertising may fail. To keep this demo simple, we don't handle failures.
        connectionsClient.startAdvertising(
            myCodeName,
            packageName,
            connectionLifecycleCallback,
            options
        ).addOnCompleteListener {
            if(it.isSuccessful){
                Log.d("WIFI","KT Advertising")
            }else if(it.exception!=null){
                Log.d("WIFI","Exception Occured Advertising:${it.exception!!.message}")
            }
        }
    }

    fun connectPeer(endpointId:String){
        connectionsClient.requestConnection(myCodeName, endpointId, connectionLifecycleCallback).addOnCompleteListener {
            if(it.isSuccessful){
                Log.d("WIFI","Requested Connection")
                connectedPeers.add(endpointId)
            }else if(it.exception!=null){
                Log.d("WIFI","Exception Occured Requesting:${it.exception!!.message}")
            }

        }
    }
    fun xstartDiscovery(){
        val options = DiscoveryOptions.Builder().setStrategy(STRATEGY).build()
        connectionsClient.startDiscovery(packageName,endpointDiscoveryCallback,options).addOnCompleteListener {
            if(it.isSuccessful){
                Log.d("WIFI","KT Discovery Successful")
            }else if(it.exception!=null){
                Log.d("WIFI","Exception Occured Discovery:${it.exception!!.message}")
            }
        }
    }


    // Callbacks for connections to other devices
    val connectionLifecycleCallback = object : ConnectionLifecycleCallback() {
        override fun onConnectionInitiated(endpointId: String, info: ConnectionInfo) {
            // Accepting a connection means you want to receive messages. Hence, the API expects
            // that you attach a PayloadCall to the acceptance


            Log.d("WIFI","Connection Accepted ${info.endpointName}")
            connectionsClient.acceptConnection(endpointId, payloadCallback)
//            opponentName = "Opponent\n(${info.endpointName})"
        }
        private val payloadCallback: PayloadCallback = object : PayloadCallback() {
            override fun onPayloadReceived(endpointId: String, payload: Payload) {
                Log.d("WIFI","PAYLOAD RECEIVED");

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
            }

            override fun onPayloadTransferUpdate(endpointId: String, update: PayloadTransferUpdate) {
                // Determines the winner and updates game state/UI after both players have chosen.
                // Feel free to refactor and extract this code into a different method

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
        // reset data
//        opponentEndpointId = null
//        opponentName = null
//        opponentChoice = null
//        opponentScore = 0
//        myChoice = null
//        myScore = 0
//        // reset state of views
//        binding.disconnect.visibility = View.GONE
//        binding.findOpponent.visibility = View.VISIBLE
//        setGameControllerEnabled(false)
//        binding.opponentName.text="opponent\n(none yet)"
//        binding.status.text ="..."
//        binding.score.text = ":"
    }



    private lateinit var connectionsClient: ConnectionsClient
    init{
        connectionsClient = Nearby.getConnectionsClient(activity)
    }
}