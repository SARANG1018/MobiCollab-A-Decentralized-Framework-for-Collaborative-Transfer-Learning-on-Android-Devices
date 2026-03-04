package com.fyp.collaborite

import android.annotation.SuppressLint
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.wifi.p2p.WifiP2pManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.CallSuper
import androidx.camera.core.ImageProxy
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ListItem
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.text.isDigitsOnly
import com.fyp.collaborite.distributed.wifi.ServiceManager
import com.fyp.collaborite.distributed.wifi.WifiKtsManager
import com.fyp.collaborite.distributed.wifi.WifiManager
import com.fyp.collaborite.ui.theme.CollaboriteTheme
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import com.google.android.gms.nearby.connection.DiscoveredEndpointInfo
import org.tensorflow.lite.examples.modelpersonalization.TransferLearningHelper
import org.tensorflow.lite.support.label.Category
import java.io.File
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class ConnectionActivity : ComponentActivity(),TransferLearningHelper.ClassifierListener {

    private val intentFilter = IntentFilter()
    lateinit var wifiManager:WifiManager
    lateinit var serviceManager: ServiceManager
    lateinit var wifiKtsManager: WifiKtsManager
    private var shouldShowCamera: MutableState<Boolean> = mutableStateOf(false)
    private var tvLossConsumerPause : MutableState<String> = mutableStateOf("")
    private var tvLossConsumerResume : MutableState<String> = mutableStateOf("")
    private var classifiedValue = mutableStateOf("--")
    private var latestBitmap: Bitmap? = null
    private var latestRotation: Int = 0
    private lateinit var outputDirectory: File
    private lateinit var cameraExecutor: ExecutorService
    private var isObjectPresent = mutableStateOf(false)
    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            ConnectionPreview()
        }

        requestCameraPermission()
        wifiKtsManager = WifiKtsManager(this)


    }
    private val REQUEST_CODE_REQUIRED_PERMISSIONS = 1

    override fun onStart() {
        super.onStart()
        if (checkSelfPermission(android.Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(
                arrayOf(android.Manifest.permission.ACCESS_FINE_LOCATION),
                REQUEST_CODE_REQUIRED_PERMISSIONS
            )
        }
    }

    override fun onResume() {
        super.onResume()
    }

    override fun onPause() {
        super.onPause()
    }


    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun AppBar(heading: String, modifier: Modifier = Modifier) {
        TopAppBar(
            title = { Text(text = heading) },
            actions = {
                androidx.compose.material3.TextButton(onClick = {
                    wifiKtsManager.startTrainingManually()
                }) {
                    Text("Train", color = androidx.compose.material3.MaterialTheme.colorScheme.primary)
                }
                androidx.compose.material3.TextButton(onClick = {
                    wifiKtsManager.syncWeightsWithPeers()
                }) {
                    Text("Sync", color = androidx.compose.material3.MaterialTheme.colorScheme.primary)
                }
                androidx.compose.material3.TextButton(onClick = {
                    val bmp = latestBitmap
                    Log.d("PIPELINE", "[TEST] Test button pressed. Has frame: ${bmp != null}")
                    if (bmp != null) {
                        wifiKtsManager.transferLearningHelper.classify(bmp, latestRotation)
                    }
                }) {
                    Text("Test", color = androidx.compose.material3.MaterialTheme.colorScheme.primary)
                }
            }
        )
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun PeerList(peers: List<String>){
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            content = {
                items(peers){it->

                    val dev: DiscoveredEndpointInfo? = wifiKtsManager.peerMap[it]

                    ListItem(
                        modifier = Modifier.fillMaxSize(),
                        headlineText = { Text(dev?.endpointName.orEmpty()) },
                        supportingText = { Text(text = dev?.serviceId.orEmpty()) },
                        trailingContent = {
                            if (!wifiKtsManager.connectedPeers.contains(it)) {
                                Button(onClick = {
                                    wifiKtsManager.connectPeer(it)
                                }) {
                                    Text("Connect")
                                }
                            } else {
                                Text("Connected")
                            }
                        })

                }
            })
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun InputField( modifier: Modifier = Modifier,label:String="",initialValue:String="",onValueChange:(it:Int)->Unit={}) {
        var textField by remember {
            mutableStateOf(initialValue)
        }

        OutlinedTextField(
            modifier= Modifier
                .fillMaxWidth()
                .padding(horizontal = 10.dp, vertical = 10.dp),
            value = textField,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            onValueChange = {
                if (it.isDigitsOnly() && it!=""){
                    textField = it;
                    onValueChange(it.toInt());
                }

            },
            label = { Text(label) }
        )

    }


    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun CButton(modifier: Modifier=Modifier,onClick: ()->Unit,text: String="Submit") {


        Button(
            onClick = onClick ,content={
                Text(text = text)
            })

    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Preview(showBackground = true)
    @Composable
    fun ConnectionPreview() {
        var data by remember {
            mutableStateOf(0)
        }
        var codeNumber by remember {
            mutableStateOf(0)
        }
        var shouldShowCamera by remember { mutableStateOf(false) }


        CollaboriteTheme {
            Scaffold (
                topBar = { AppBar(heading = "Connection") }
            ){
                Column(modifier=Modifier.padding(it), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(text = "Image Count:${wifiKtsManager.sampleCount.value}")
                    Text(text = "Training")
                    Text(text = "Loss:")
                    Text(text = tvLossConsumerPause.value)
                    Text(text = "Classified Value: ${classifiedValue.value}")

                    InputField(label = "Start Discovery", onValueChange = {
                        codeNumber = it
                    })
                    Button(
                        enabled = codeNumber != 0,
                        content = { Text(text = "submit") },
                        onClick = {
                            wifiKtsManager.xstartAdvertising(codeNumber.toString())
                            wifiKtsManager.xstartDiscovery()
                        })
                    Text("______")
                    Box{
                        if (shouldShowCamera) {
                            CameraView(
                                outputDirectory = outputDirectory,
                                executor = cameraExecutor,
                                onTrueImage = ::handleImageCapture,
                                onFalseImage= ::handleImageCapture,

                                onError = { Log.e("kilo", "View error:", it) },
                                onImageUpdate = { bmp, rot ->
                                    latestBitmap = bmp
                                    latestRotation = rot
                                }
                            )
                        } else {
                            Column {
                                Text(text = "Peers")
                                PeerList(peers = wifiKtsManager.peers)
                            }
                        }

                        IconButton(
                            enabled = wifiKtsManager.peers.isNotEmpty() || shouldShowCamera,
                            modifier = Modifier.align(Alignment.TopEnd),
                            onClick = {
                                shouldShowCamera = !shouldShowCamera
                            }) {
                            Icon(
                                modifier = Modifier.padding(8.dp),
                                tint = if (shouldShowCamera) Color.White else Color.Black,
                                imageVector = Icons.Filled.ArrowBack,
                                contentDescription = "Back"
                            )
                        }
                    }
                }
            }
        }
    }

    private fun requestCameraPermission() {

        outputDirectory = getOutputDirectory()
        cameraExecutor = Executors.newSingleThreadExecutor()
        when {
            ContextCompat.checkSelfPermission(
                this,
                android.Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                shouldShowCamera.value = true
            }
            ActivityCompat.shouldShowRequestPermissionRationale(
                this,
                android.Manifest.permission.CAMERA
            ) -> Log.i("kilo", "Show camera permissions dialog")

            else -> requestPermissionLauncher.launch(android.Manifest.permission.CAMERA)
        }
    }
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            shouldShowCamera.value = true
        } else {
            Log.i("kilo", "Permission denied")
        }
    }
    fun getResizedBitmap(image: Bitmap, maxSize: Int): Bitmap? {
        var width = image.width
        var height = image.height
        val bitmapRatio = width.toFloat() / height.toFloat()
        if (bitmapRatio > 0) {
            width = maxSize
            height = (width / bitmapRatio).toInt()
        } else {
            height = maxSize
            width = (height * bitmapRatio).toInt()
        }
        return Bitmap.createScaledBitmap(image, width, height, true)
    }
    private fun handleImageCapture(image: ImageProxy,className:String) {
        Log.d("PIPELINE", "[CAPTURE] Image captured for class=$className")

        val imageRotation = image.imageInfo.rotationDegrees
        val buffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())

        buffer.get(bytes)
        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        Log.d("PIPELINE", "[CAPTURE] Bitmap decoded: ${bitmap.width}x${bitmap.height}, rotation=$imageRotation")
        
        wifiKtsManager.transferLearningHelper.addSample(bitmap, className, imageRotation)
        wifiKtsManager.sampleCount.value = wifiKtsManager.transferLearningHelper.getSampleCount()
        Log.d("PIPELINE", "[CAPTURE] Sample added. Total samples: ${wifiKtsManager.sampleCount.value}")
        image.close()
    }

    private fun getOutputDirectory(): File {
        val mediaDir = externalMediaDirs.firstOrNull()?.let {
            File(it, resources.getString(R.string.app_name)).apply { mkdirs() }
        }

        return if (mediaDir != null && mediaDir.exists()) mediaDir else filesDir
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }


    @SuppressLint("NotifyDataSetChanged")
    override fun onError(error: String) {
        Toast.makeText(applicationContext, error, Toast.LENGTH_SHORT).show()
    }

    @SuppressLint("NotifyDataSetChanged")
    override fun onResults(
        results: List<Category>?,
        inferenceTime: Long
    ) {
        results?.let { list ->
            val top = list.maxByOrNull { it.score }
            if (top != null) {
                classifiedValue.value = top.label
            }
            Log.d("PIPELINE", "[TEST] Classification result: ${list.joinToString { "${it.label}=${String.format(Locale.US, "%.3f", it.score)}" }}")
            Log.d("PIPELINE", "[TEST] Top prediction: ${top?.label}")
        }
    }

    override fun onLossResults(lossNumber: Float) {
        Log.d("PIPELINE", "[TRAIN] Loss: $lossNumber")
        String.format(
            Locale.US,
            "Loss: %.3f", lossNumber
        ).let {
            tvLossConsumerPause.value = it
            tvLossConsumerResume.value = it
        }
    }
}