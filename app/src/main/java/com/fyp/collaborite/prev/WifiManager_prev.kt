package com.fyp.collaborite.distributed.wifi

import android.Manifest
import android.annotation.SuppressLint
import android.content.BroadcastReceiver
import android.content.Context
import android.content.pm.PackageManager
import android.net.wifi.WpsInfo
import android.net.wifi.p2p.WifiP2pConfig
import android.net.wifi.p2p.WifiP2pDevice
import android.net.wifi.p2p.WifiP2pGroup
import android.net.wifi.p2p.WifiP2pManager
import android.util.Log
import android.widget.ListAdapter
import android.widget.Toast
import androidx.compose.runtime.Composable
import androidx.compose.runtime.State
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.core.app.ActivityCompat
import androidx.recyclerview.widget.AsyncListDiffer.ListListener
import com.fyp.collaborite.ConnectionActivity

class WifiManager(val activity: ConnectionActivity,val manager: WifiP2pManager,val channel: WifiP2pManager.Channel) {
    fun registerReceiver():WifiDirectBroadcastReceiver?{
        receiver = WifiDirectBroadcastReceiver(manager, channel, this.activity)
        return  receiver
    }


    var isWifiP2pEnabled: Boolean=false
    var receiver:  WifiDirectBroadcastReceiver? = null
    private val peers = mutableListOf<WifiP2pDevice>()

    private val peerListListener = WifiP2pManager.PeerListListener { peerList ->
        val refreshedPeers = peerList.deviceList
        if (refreshedPeers != peers) {
            peers.clear()
            peers.addAll(refreshedPeers)

            Log.d("WIFI","Peers Discovered${peers.size}")


            // If an AdapterView is backed by this data, notify it
            // of the change. For instance, if you have a ListView of
            // available peers, trigger an update.
//            (listAdapter as WiFiPeerListAdapter).notifyDataSetChanged()

            // Perform any other updates needed based on the new list of
            // peers connected to the Wi-Fi P2P network.
        }

        if (peers.isEmpty()) {
            Log.d("PEER", "No devices found")
            return@PeerListListener
        }
    }

    @SuppressLint("MissingPermission")
    fun discoverPeers(){

        manager.discoverPeers(channel, object : WifiP2pManager.ActionListener {

            override fun onSuccess() {
                // Code for when the discovery initiation is successful goes here.
                // No services have actually been discovered yet, so this method
                // can often be left blank. Code for peer discovery goes in the
                // onReceive method, detailed below.
                Log.d("WIFI","Discovery Started")
            }

            override fun onFailure(reasonCode: Int) {
                // Code for when the discovery initiation fails goes here.
                // Alert the user that something went wrong.
                WifiP2pManager.CONNECTION_REQUEST_DEFER_TO_SERVICE


                Log.d("WIFI","Discovery Initialization Failed ${reasonCode}")
            }
        })
    }

    @SuppressLint("MissingPermission")
    fun requestPeers(){
        manager.requestPeers(channel, peerListListener)
        Log.d("REQUESTED FOR PEERS", "P2P peers changed")
    }


    @SuppressLint("MissingPermission")
    fun connectPeer(device: WifiP2pDevice){

        val config = WifiP2pConfig().apply {
            deviceAddress = device.deviceAddress
            wps.setup = WpsInfo.PBC
        }
        manager.connect(channel,config,
            object :WifiP2pManager.ActionListener{
                override fun onSuccess() {
                    TODO("Not yet implemented")
                }

                override fun onFailure(p0: Int) {
                    Toast.makeText(
                        activity,
                        "Connect failed. Retry.",
                        Toast.LENGTH_SHORT
                    ).show()
                }

            })
    }

    @SuppressLint("MissingPermission")
    fun createGroup(){
        manager.createGroup(channel, object : WifiP2pManager.ActionListener {
            override fun onSuccess() {
                // Device is ready to accept incoming connections from peers.
                manager.requestGroupInfo(channel,object :WifiP2pManager.GroupInfoListener{
                    override fun onGroupInfoAvailable(p0: WifiP2pGroup?) {
                        Log.d("WIFI",p0.toString())
                    }
                })
            }

            override fun onFailure(reason: Int) {

                Toast.makeText(
                    activity,
                    "P2P group creation failed. Retry.${reason}",
                    Toast.LENGTH_SHORT
                ).show()
            }
        })
    }


}