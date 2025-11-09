package com.example.resistorvision

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.example.resistorvision.ui.theme.ResistorVisionTheme
import java.nio.FloatBuffer
import java.util.Collections

class MainActivity : ComponentActivity() {

    private val cameraPermissionRequest =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                // Permission granted, you can now use the camera
            } else {
                // Permission denied. You might want to show a message to the user.
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            cameraPermissionRequest.launch(Manifest.permission.CAMERA)
        }
        setContent {
            ResistorVisionTheme {
                val modelOutput = remember { mutableStateOf<String?>(null) }
                AppContent(modelOutput)
            }
        }
    }
}

@Composable
fun AppContent(modelOutput: MutableState<String?>) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraController = remember { LifecycleCameraController(context) }
    val objectDetector = remember { ObjectDetector(context) }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                PreviewView(ctx).apply {
                    this.controller = cameraController
                    cameraController.bindToLifecycle(lifecycleOwner)
                    cameraController.cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                }
            }
        )
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.Bottom,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Button(
                onClick = {
                    captureAndProcessImage(context, cameraController, objectDetector, modelOutput)
                },
                modifier = Modifier.padding(16.dp)
            ) {
                Text(text = "Capture")
            }
            modelOutput.value?.let {
                Text(
                    text = "Detected Colors: \n$it",
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    color = Color.White,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

private fun captureAndProcessImage(
    context: Context,
    cameraController: CameraController,
    objectDetector: ObjectDetector,
    modelOutput: MutableState<String?>
) {
    cameraController.takePicture(
        ContextCompat.getMainExecutor(context),
        object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                super.onCaptureSuccess(image)
                val bitmap = imageProxyToBitmap(image)
                image.close()

                val result = objectDetector.detect(bitmap)
                modelOutput.value = result.joinToString(", ")
            }

            override fun onError(exception: ImageCaptureException) {
                super.onError(exception)
                Log.e("Camera", "Error capturing image", exception)
                modelOutput.value = "Error capturing image."
            }
        }
    )
}

private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
    val planeProxy = image.planes[0]
    val buffer = planeProxy.buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)
    val initialBitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)

    val matrix = Matrix().apply {
        postRotate(image.imageInfo.rotationDegrees.toFloat())
    }
    return Bitmap.createBitmap(initialBitmap, 0, 0, initialBitmap.width, initialBitmap.height, matrix, true)
}

class ObjectDetector(context: Context) {
    private val classNames = listOf(
        "black", "blue", "brown", "gold", "green", "grey", "orange", "red", "silver", "violet", "white", "yellow"
    )
    private var ortSession: OrtSession?
    private val ortEnvironment = OrtEnvironment.getEnvironment()

    init {
        try {
            val modelBytes = context.assets.open("best.onnx").readBytes()
            ortSession = ortEnvironment.createSession(modelBytes)
        } catch (e: Exception) { // This now catches OrtException, IOException, etc.
            Log.e(
                "ObjectDetector",
                "FATAL: Failed to initialize ONNX model. Check logs for OrtException, likely due to unsupported opset.",
                e
            )
            ortSession = null
        }
    }

    fun detect(bitmap: Bitmap): List<String> {
        if (ortSession == null) {
            return listOf("Error: Model failed to load")
        }

        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
        val inputTensor = bitmapToTensor(resizedBitmap)
        val inputName = ortSession?.inputNames?.first()
        val inputs = Collections.singletonMap(inputName, inputTensor)

        try {
            val results = ortSession?.run(inputs)
            val output = results?.get(0)?.value as? Array<FloatArray>

            return if (output != null) {
                processOutput(output)
            } else {
                Log.e("ObjectDetector", "Model output was null or of an unexpected type.")
                listOf("Error: Invalid model output")
            }
        } catch (e: Exception) {
            Log.e("ObjectDetector", "Error during model inference.", e)
            return listOf("Error: Inference failed")
        }
    }

    private fun bitmapToTensor(bitmap: Bitmap): OnnxTensor {
        val floatBuffer = FloatBuffer.allocate(1 * 3 * 640 * 640)
        floatBuffer.rewind()
        val pixels = IntArray(640 * 640)
        bitmap.getPixels(pixels, 0, 640, 0, 0, 640, 640)

        for (pixelValue in pixels) {
            val r = ((pixelValue shr 16) and 0xFF) / 255.0f
            val g = ((pixelValue shr 8) and 0xFF) / 255.0f
            val b = (pixelValue and 0xFF) / 255.0f
            floatBuffer.put(r)
            floatBuffer.put(g)
            floatBuffer.put(b)
        }

        floatBuffer.rewind()
        val shape = longArrayOf(1, 3, 640, 640)
        return OnnxTensor.createTensor(ortEnvironment, floatBuffer, shape)
    }

    private fun processOutput(output: Array<FloatArray>): List<String> {
        val detections = mutableListOf<Pair<Float, String>>()

        output.forEach { detection ->
            val confidence = detection[4]
            if (confidence > 0.25) {
                val x1 = detection[0]
                val classId = detection[5].toInt()
                if (classId in classNames.indices) {
                    detections.add(Pair(x1, classNames[classId]))
                }
            }
        }

        return detections.sortedBy { it.first }.map { it.second }
    }
}
@Preview(showBackground = true, showSystemUi = true)
@Composable
fun AppContentPreview() {
    ResistorVisionTheme {
        // We use a fake state for the preview, as the real one requires the camera.
        val fakeModelOutput = remember { mutableStateOf("brown, black, orange, gold") }

        // We can't show the real camera in a preview, so we'll simulate the UI.
        Box(modifier = Modifier.fillMaxSize()) {
            // A simple black background to simulate the camera view.
            Box(modifier = Modifier.fillMaxSize().background(Color.Black))

            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp),
                verticalArrangement = Arrangement.Bottom,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Button(
                    onClick = { /* In preview, this does nothing */ },
                    modifier = Modifier.padding(16.dp)
                ) {
                    Text(text = "Capture")
                }

                // Display the fake output text so we can see how it looks.
                fakeModelOutput.value?.let {
                    Text(
                        text = "Detected Colors: \n$it",
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        color = Color.White,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }
    }
}
