package com.fyp.collaborite.components

import android.app.Activity
import android.app.AppComponentFactory
import android.content.Context
import android.content.pm.PackageManager
import androidx.activity.result.contract.ActivityResultContract
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ListItem
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.core.app.ActivityCompat

class Components {


}


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PermissionList(permissions: Array<String>, applicationContext: Context,currentActivity: Activity){

    LazyColumn(

        content = {
            items(permissions){
                ListItem(headlineText = {
                    Text(it)
                }, trailingContent = {
                    if(ActivityCompat.checkSelfPermission(applicationContext,it)== PackageManager.PERMISSION_GRANTED){
                        Text("Granted")
                    }else{
                        Button(onClick = {

                        }) {
                            Text("Grant Permission")

                        }
                    }
                })

            }
        })
}