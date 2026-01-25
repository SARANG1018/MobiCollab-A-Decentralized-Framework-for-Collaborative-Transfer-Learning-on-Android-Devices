package com.fyp.collaborite

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.database.DataSetObserver
import android.net.wifi.WpsInfo
import android.net.wifi.p2p.WifiP2pConfig
import android.net.wifi.p2p.WifiP2pDevice
import android.net.wifi.p2p.WifiP2pDeviceList
import android.net.wifi.p2p.WifiP2pManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.widget.ListAdapter
import android.widget.ListView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.core.app.ActivityCompat
import androidx.core.widget.ListViewCompat
import com.fyp.collaborite.distributed.wifi.WifiDirectBroadcastReceiver
import com.fyp.collaborite.ui.theme.CollaboriteTheme
import org.w3c.dom.Text
import java.io.IOException
import java.io.Serializable
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket


class WifiActivity : ComponentActivity(){
    private lateinit var manager: WifiP2pManager

    var data= mutableStateOf("")
    var responseData= mutableStateOf("")



    val socket = Socket()
    val buf = ByteArray(1024)


//    var peers= arrayOf(mapOf("name" to "Tanmai","device" to "123"),mapOf("name" to "Kiran","device" to "123"));

    var channel: WifiP2pManager.Channel? = null
    var receiver:  WifiDirectBroadcastReceiver? = null
    val intentFilter = IntentFilter().apply {
        addAction(WifiP2pManager.WIFI_P2P_STATE_CHANGED_ACTION)
        addAction(WifiP2pManager.WIFI_P2P_PEERS_CHANGED_ACTION)
        addAction(WifiP2pManager.WIFI_P2P_CONNECTION_CHANGED_ACTION)
        addAction(WifiP2pManager.WIFI_P2P_THIS_DEVICE_CHANGED_ACTION)
    }


    fun discoverPeers(){
        if (  ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.NEARBY_WIFI_DEVICES
            ) != PackageManager.PERMISSION_GRANTED
        ) {

            Log.d("WIFI","Permissions Denied")
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
//               public void onRequestPermissionsResult(int requestCode, String[] permissions,
//                                                      int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
//            ActivityCompat.requestPermissions(this,Array<String>(2,{Manifest.permission.L}),100)
            return
        }
        manager.discoverPeers(channel, object : WifiP2pManager.ActionListener {

            override fun onSuccess() {
                // Code for when the discovery initiation is successful goes here.
                // No services have actually been discovered yet, so this method
                // can often be left blank. Code for peer discovery goes in the
                // onReceive method, detailed below.
            }


            override fun onFailure(reasonCode: Int) {
                // Code for when the discovery initiation fails goes here.
                // Alert the user that something went wrong.
            }
        })
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        manager = getSystemService(Context.WIFI_P2P_SERVICE) as WifiP2pManager
        channel = manager?.initialize(this, mainLooper, null)
        channel?.also { channel ->
//            receiver = WifiDirectBroadcastReceiver(manager, channel, this)
        }



        val permisson= arrayOf(Manifest.permission.ACCESS_WIFI_STATE,Manifest.permission.ACCESS_FINE_LOCATION,Manifest.permission.NEARBY_WIFI_DEVICES)
        setContent {
            CollaboriteTheme {


                Column {
                    Text(responseData.value)
                    Text("Wifi Channel:${channel}")
                    Text("Manager:${manager}")
                    Button(onClick = { discoverPeers() }) {
                        Text("Discover")
                    }
                    PermissionList(permissions = permisson)

                    Text("PeerList")
                    PeerList(peers =receiver?.peers.orEmpty())

                }

            }
        }
    }

    /* register the broadcast receiver with the intent values to be matched */
    override fun onResume() {
        super.onResume()
        receiver?.also { receiver ->
            registerReceiver(receiver, intentFilter)
        }
    }

    /* unregister the broadcast receiver */
    override fun onPause() {
        super.onPause()
        receiver?.also { receiver ->
            unregisterReceiver(receiver)
        }
    }

    fun requestPermission(permisssion:String){
        ActivityCompat.requestPermissions(this, arrayOf(permisssion),100)
    }

    fun connectDevice(device : WifiP2pDevice){
        val config = WifiP2pConfig().apply {
            deviceAddress = device.deviceAddress
            wps.setup = WpsInfo.PBC
        }

        if ( ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.NEARBY_WIFI_DEVICES
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            Log.d("WIFI", "No Permissions")
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            return
        }
        manager.connect(channel, config, object : WifiP2pManager.ActionListener {

            override fun onSuccess() {

                // WiFiDirectBroadcastReceiver notifies us. Ignore for now.

            }


            override fun onFailure(reason: Int) {
                Toast.makeText(
                    this@WifiActivity,
                    "Connect failed. Retry.",
                    Toast.LENGTH_SHORT
                ).show()
            }
        })
    }

    fun executeTemp(it : WifiP2pDevice){
        Log.d("SERVER", "Starting")


        Log.d("SERVER", "Socket Initialized")
        if(it.deviceName=="realme 9 Pro+"){

            Log.d("SERVER", "connecting Server")

            Thread{
                val host: String=it.deviceAddress
                val port: Int=8888
                try {
                    /**
                     * Create a client socket with the host,
                     * port, and timeout information.
                     */
                    socket.bind(null)
                    socket.connect((InetSocketAddress(host, port)), 50000)

                    Log.d("SERVER", "Server Connected")

                    /**
                     * Create a byte stream from a JPEG file and pipe it to the output stream
                     * of the socket. This data is retrieved by the server device.
                     */
                    val outputStream = socket.getOutputStream()
                    val inputStream = socket.getInputStream()
                    val data=inputStream.read()
                    outputStream.write(data*data)
                    outputStream.close()
                    inputStream.close()
                } catch (e: IOException) {
                    //catch logic
                    Log.d("SERVER",e.toString())
                } finally {
                    /**
                     * Clean up any open sockets when done
                     * transferring or if an exception occurred.
                     */
                    socket.takeIf { it.isConnected }?.apply {
                        close()
                    }
                }
            }.start()
        }else{
            Thread{
                val serverSocket = ServerSocket(8888)
                Log.d("SERVER", "Starting Server")
                serverSocket.use {
                    /**
                     * Wait for client connections. This call blocks until a
                     * connection is accepted from a client.
                     */
                    val client = serverSocket!!.accept()

                    Log.d("SERVER", "Client Connected")
                    /**
                     * If this code is reached, a client has connected and transferred data
                     * Save the input stream from the client as a JPEG file
                     */
//            val f = File(Environment.getExternalStorageDirectory().absolutePath +
//                    "/${context.packageName}/wifip2pshared-${System.currentTimeMillis()}.jpg")
//            val dirs = File(f.parent)
//
//            dirs.takeIf { it.doesNotExist() }?.apply {
//                mkdirs()
//            }
//            f.createNewFile()
                    val outputStream=client.getOutputStream();
                    outputStream.write(10);
                    val inputStream=client.getInputStream()
                    val ans=inputStream.read()
                    Log.d("OUTPUT",ans.toString())
                    serverSocket!!.close()
                    responseData.value=ans.toString();
                }
            }.start()
        }
    }
    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun PeerList(peers: List<WifiP2pDevice>){
        LazyColumn(

            modifier = Modifier.fillMaxSize(),
            content = {
                items(peers){it->
                    androidx.compose.material3.ListItem(
                        modifier = Modifier.fillMaxSize(),
                        headlineText = { Text(it.deviceName.orEmpty()) },
                        supportingText = {
                            Text(text =it.deviceAddress.orEmpty()
                            )}, trailingContent = {
                            if(it.status!=WifiP2pDevice.CONNECTED){
//                                executeTemp(it)


                                Button(onClick = {
                                    connectDevice(it)

                                }) {
                                    Text("Connect")
                                }
                            }else{

                                Text("Connected")


                            }
                        })

                }
            })
    }


    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun PermissionList(permissions: Array<String>){
        LazyColumn(

            content = {
            items(permissions){
                androidx.compose.material3.ListItem(headlineText = {
                    Text(it)
                }, trailingContent = {
                    if(ActivityCompat.checkSelfPermission(applicationContext,it)==PackageManager.PERMISSION_GRANTED){
                        Text("Granted")
                    }else{
                        Button(onClick = {

                            requestPermission(it)
                        }) {
                            Text("Grant Permission")

                        }
                    }
                })

            }
        })
    }
    @Preview(showBackground = true)
    @Composable
    fun DefaultPreview(){
        val permisson= arrayOf(Manifest.permission.ACCESS_WIFI_STATE)
        CollaboriteTheme {
            PermissionList(permisson)


//            Column {
//                Text("Wifi Channel:${channel}")
//                Text("Manager:${manager}")
//                PeerList(peers = peers)
//            }

        }
    }



}


