//package com.fyp.collaborite.distributed.wifi
//
//import android.os.Handler
//import android.os.Looper
//import android.widget.TextView
//import java.util.concurrent.Callable
//import java.util.concurrent.Executor
//import java.util.concurrent.Executors
//
//
//class TaskRunner {
//    private val executor: Executor =
//        Executors.newSingleThreadExecutor() // change according to your requirements
//    private val handler: Handler = Handler(Looper.getMainLooper())
//
//    class Callback<R> {
//        fun onComplete(result: R){
//
//        }
//    }
//
//    fun <R> executeAsync(callable: Callable<String>, callback: () -> Callback<String>) {
//        executor.execute {
//            val result: String = callable.call()
//            handler.post { callback.onComplete(result) }
//        }
//    }
//}
//class FileServerAsyncTask(
//
//    private var statusText: TextView
//) {
//
//     fun doInBackground(vararg params: Void){
//        Thread{
//
//            /**
//             * Create a server socket.
//             */
//
//        }.start()
//    }
//
//}