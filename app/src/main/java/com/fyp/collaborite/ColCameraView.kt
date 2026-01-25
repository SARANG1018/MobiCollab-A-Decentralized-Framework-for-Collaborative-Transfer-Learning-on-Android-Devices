package com.fyp.collaborite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.sharp.Add
import androidx.compose.material.icons.sharp.Clear
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.Executor
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine


//class ColCameraView{
    private suspend fun Context.getCameraProvider(): ProcessCameraProvider = suspendCoroutine { continuation ->
        ProcessCameraProvider.getInstance(this).also { cameraProvider ->
            cameraProvider.addListener({
                continuation.resume(cameraProvider.get())
            }, ContextCompat.getMainExecutor(this))
        }
    }

    @Composable
    public fun CameraView(
        outputDirectory: File,
        executor: Executor,
        onTrueImage: (ImageProxy,String) -> Unit,
        onFalseImage: (ImageProxy,String) -> Unit,
        onError: (ImageCaptureException) -> Unit,
        onImageUpdate: (Bitmap,Int)->Unit
    ) {
        // 1
        val lensFacing = CameraSelector.LENS_FACING_BACK
        val context = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current


        val preview = Preview.Builder().build()
        val previewView = remember { PreviewView(context) }
        val imageCapture: ImageCapture = remember { ImageCapture.Builder().build() }
        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(lensFacing)
            .build()

        // 2
        LaunchedEffect(lensFacing) {
            val cameraProvider = context.getCameraProvider()
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageCapture
            )

            preview.setSurfaceProvider(previewView.surfaceProvider)
        }

        // 3
        Box(contentAlignment = Alignment.BottomCenter, modifier = Modifier.fillMaxSize()) {
            AndroidView({ previewView }, modifier = Modifier.fillMaxSize(), update = {

                it.rotation
                it.bitmap.let {x->
                    if(x!=null){
                        onImageUpdate(x,it.rotation.toInt())

                    }
                }

            })

           Row {
               IconButton(
                   modifier = Modifier.padding(bottom = 20.dp),
                   onClick = {
                       Log.i("kilo", "ON CLICK")
                       takePhotoAndSaveAsBitmap(

                           imageCapture = imageCapture,
                           executor = executor,
                           onBitmapCaptured = onTrueImage,
                           onError = onError,
                           className = "1"
                       )
                   },
                   content = {
                       Icon(
                           imageVector = Icons.Sharp.Add,
                           contentDescription = "Take picture",
                           tint = Color.White,
                           modifier = Modifier
                               .size(100.dp)
                               .padding(1.dp)
                               .border(1.dp, Color.White, CircleShape)
                       )
                   }
               )
               IconButton(
                   modifier = Modifier.padding(bottom = 20.dp),
                   onClick = {
                       Log.i("kilo", "ON CLICK")
                       takePhotoAndSaveAsBitmap(

                           imageCapture = imageCapture,
                           executor = executor,
                           onBitmapCaptured = onFalseImage,
                           onError = onError,
                           className = "2"
                       )
                   },
                   content = {
                       Icon(
                           imageVector = Icons.Sharp.Clear,
                           contentDescription = "Take picture",
                           tint = Color.White,
                           modifier = Modifier
                               .size(100.dp)
                               .padding(1.dp)
                               .border(1.dp, Color.White, CircleShape)
                       )
                   }
               )
           }
        }
    }

private fun takePhotoAndSaveAsBitmap(
    imageCapture: ImageCapture,
    executor: Executor,
    onBitmapCaptured: (ImageProxy,String) -> Unit,
    onError: (ImageCaptureException) -> Unit,
    className: String
) {
    // Capture the image and process it
    imageCapture.takePicture(executor, object : ImageCapture.OnImageCapturedCallback() {
        override fun onError(exception: ImageCaptureException) {
            // Handle capture error
            onError(exception)
        }

        override fun onCaptureSuccess(image: ImageProxy) {
            // Convert ImageProxy to Bitmap


            // Call the callback with the captured bitmap

            onBitmapCaptured(image,className)
        }
    })
}

    private fun takePhoto(
        filenameFormat: String,
        imageCapture: ImageCapture,
        outputDirectory: File,
        executor: Executor,
        onImageCaptured: (Uri) -> Unit,
        onError: (ImageCaptureException) -> Unit
    ) {

        val photoFile = File(
            outputDirectory,
            SimpleDateFormat(filenameFormat, Locale.US).format(System.currentTimeMillis()) + ".jpg"
        )

        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        imageCapture.takePicture(outputOptions, executor, object: ImageCapture.OnImageSavedCallback {
            override fun onError(exception: ImageCaptureException) {
                Log.e("kilo", "Take photo error:", exception)
                onError(exception)
            }

            override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                val savedUri = Uri.fromFile(photoFile)
                onImageCaptured(savedUri)
            }
        })
    }
//}