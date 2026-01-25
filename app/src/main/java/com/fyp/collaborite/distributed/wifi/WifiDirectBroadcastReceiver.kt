package com.fyp.collaborite.distributed.wifi

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.net.wifi.p2p.WifiP2pDevice
import android.net.wifi.p2p.WifiP2pManager
import android.util.Log
import androidx.compose.runtime.mutableStateListOf
import com.fyp.collaborite.ConnectionActivity
import com.fyp.collaborite.WifiActivity

/**
 * A BroadcastReceiver that notifies of important Wi-Fi p2p events.
 */
class WifiDirectBroadcastReceiver(
    private val manager: WifiP2pManager?,
    private val channel: WifiP2pManager.Channel,
    private val activity: ConnectionActivity
) : BroadcastReceiver() {

    val peers= mutableStateListOf<WifiP2pDevice>()
    private val peerListListener = WifiP2pManager.PeerListListener { peerList ->
        val refreshedPeers = peerList.deviceList
        if (refreshedPeers != peers) {
            peers.clear()
            peers.addAll(refreshedPeers)



            // If an AdapterView is backed by this data, notify it
            // of the change. For instance, if you have a ListView of
            // available peers, trigger an update.


            // Perform any other updates needed based on the new list of
            // peers connected to the Wi-Fi P2P network.
        }

        if (peers.isEmpty()) {
            Log.d("WIFI", "No devices found")
            return@PeerListListener
        }
    }




    override fun onReceive(context: Context, intent: Intent) {

        when(intent.action) {
            WifiP2pManager.WIFI_P2P_STATE_CHANGED_ACTION -> {
                // Determine if Wi-Fi Direct mode is enabled or not, alert
                // the Activity.
                val state = intent.getIntExtra(WifiP2pManager.EXTRA_WIFI_STATE, -1)
                activity.wifiManager.isWifiP2pEnabled = state == WifiP2pManager.WIFI_P2P_STATE_ENABLED
                Log.d("WIFI","State Change:$state")

            }
            WifiP2pManager.WIFI_P2P_PEERS_CHANGED_ACTION -> {
              activity.wifiManager.requestPeers();
                Log.d("WIFI","Peers Changed")


            }
            WifiP2pManager.WIFI_P2P_CONNECTION_CHANGED_ACTION -> {
                Log.d("WIFI","Connection Changed")

                // Connection state changed! We should probably do something about
                // that.

            }
            WifiP2pManager.WIFI_P2P_THIS_DEVICE_CHANGED_ACTION -> {
                Log.d("WIFI","Device Changed")

            }
        }
    }
}