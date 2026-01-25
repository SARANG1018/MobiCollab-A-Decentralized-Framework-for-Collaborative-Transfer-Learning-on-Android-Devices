package com.fyp.collaborite.distributed.wifi

import android.annotation.SuppressLint
import android.net.wifi.p2p.WifiP2pManager
import android.net.wifi.p2p.nsd.WifiP2pDnsSdServiceInfo
import android.util.Log

class ServiceManager(val wifiManager: WifiManager) {
    val SERVER_PORT=8001
    @SuppressLint("MissingPermission")
    fun startRegistration() {
        //  Create a string map containing information about your service.
        val record: Map<String, String> = mapOf(
            "listenport" to SERVER_PORT.toString(),
            "buddyname" to "John Doe${(Math.random() * 1000).toInt()}",
            "available" to "visible"
        )

        // Service information.  Pass it an instance name, service type
        // _protocol._transportlayer , and the map containing
        // information other devices will want once they connect to this one.
        val serviceInfo =
            WifiP2pDnsSdServiceInfo.newInstance("_test", "_presence._tcp", record)

        // Add the local service, sending the service info, network channel,
        // and listener that will be used to indicate success or failure of
        // the request.
        wifiManager.manager.addLocalService(wifiManager.channel, serviceInfo, object : WifiP2pManager.ActionListener {
            override fun onSuccess() {
                Log.d("WIFI","Service Started on ${SERVER_PORT}")
                // Command successful! Code isn't necessarily needed here,
                // Unless you want to update the UI or add logging statements.
            }

            override fun onFailure(arg0: Int) {
                // Command failed.  Check for P2P_UNSUPPORTED, ERROR, or BUSY

                Log.d("WIFI","Error is Startng")
            }
        })
    }
    private val buddies = mutableMapOf<String, String>()

    @SuppressLint("MissingPermission")
    fun discoverServices(){
        wifiManager.manager.discoverServices(
            wifiManager.channel,
            object : WifiP2pManager.ActionListener {
                override fun onSuccess() {
                    Log.d("WIFI","Found Serives")
                    // Success!
                }

                override fun onFailure(code: Int) {
                    // Command failed. Check for P2P_UNSUPPORTED, ERROR, or BUSY
                    Log.d("WIFI", "Error.${code}")
                    when (code) {
                        WifiP2pManager.P2P_UNSUPPORTED -> {
                            Log.d("WIFI", "Wi-Fi Direct isn't supported on this device.")
                        }
                    }
                }
            }
        )
    }
}