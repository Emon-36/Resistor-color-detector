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
    val cameraController = remember {
        LifecycleCameraController(context).apply {
            // Set the default zoom ratio to 1.5x
            setZoomRatio(1.5f)
        }
    }
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
                        .background(Color.Black.copy(alpha = 0.5f)) // Add semi-transparent background for readability
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
                modelOutput.value = if (result.isNotEmpty()) result.joinToString(", ") else "No colors detected."
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

// Data class to hold detection information, making the code cleaner
data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float,
    val classId: Int,
    val className: String
)

class ObjectDetector(context: Context) {
    private val classNames = listOf(
        "black", "blue", "brown", "gold", "green", "grey", "orange", "red", "silver", "violet", "white", "yellow"
    )
    private var ortSession: OrtSession?
    private val ortEnvironment = OrtEnvironment.getEnvironment()

    // --- NMS Configuration ---
    private val confidenceThreshold = 0.25f // Confidence threshold to filter weak detections
    private val iouThreshold = 0.45f // IoU threshold to filter overlapping boxes

    init {
        try {
            val modelBytes = context.assets.open("best.onnx").readBytes()
            ortSession = ortEnvironment.createSession(modelBytes)
        } catch (e: Exception) {
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
                processOutputWithNMS(output) // Call the new NMS function
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

    /**
     * Calculates the Intersection over Union (IoU) between two bounding boxes.
     */
    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)

        val intersectionArea = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        val box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        val unionArea = box1Area + box2Area - intersectionArea

        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }

    /**
     * Processes the raw model output, applying Non-Maximum Suppression and then
     * a final deduplication step to produce the clean, final sequence.
     */
    private fun processOutputWithNMS(output: Array<FloatArray>): List<String> {
        val allBoxes = mutableListOf<BoundingBox>()

        // Step 1: Decode all raw detections into BoundingBox objects
        output.forEach { detection ->
            val confidence = detection[4]
            if (confidence >= confidenceThreshold) {
                val classId = detection[5].toInt()
                if (classId in classNames.indices) {
                    allBoxes.add(
                        BoundingBox(
                            x1 = detection[0],
                            y1 = detection[1],
                            x2 = detection[2],
                            y2 = detection[3],
                            confidence = confidence,
                            classId = classId,
                            className = classNames[classId]
                        )
                    )
                }
            }
        }

        if (allBoxes.isEmpty()) {
            return emptyList()
        }

        // Step 2: Apply NMS for each class separately
        val finalDetections = mutableListOf<BoundingBox>()
        val groupedBoxes = allBoxes.groupBy { it.classId }

        groupedBoxes.forEach { (_, boxes) ->
            var remainingBoxes = boxes.sortedByDescending { it.confidence }.toMutableList()

            while (remainingBoxes.isNotEmpty()) {
                val bestBox = remainingBoxes.first()
                finalDetections.add(bestBox) // Keep the box with the highest confidence
                remainingBoxes.removeAt(0) // FIX: Use removeAt(0) for API compatibility

                // Remove all other boxes that have a high IoU with the best box
                remainingBoxes = remainingBoxes.filter { box ->
                    calculateIoU(bestBox, box) < iouThreshold
                }.toMutableList()
            }
        }

        // Step 3: Sort the clean detections by their horizontal position
        val sortedColors = finalDetections.sortedBy { it.x1 }.map { it.className }

        // Step 4 (NEW): Deduplicate the final sorted list
        // This removes consecutive identical colors (e.g., [brown, brown, black] -> [brown, black])
        if (sortedColors.isEmpty()) {
            return emptyList()
        }

        val deduplicatedColors = mutableListOf<String>()
        deduplicatedColors.add(sortedColors.first()) // Add the very first color

        for (i in 1 until sortedColors.size) {
            // Only add the current color if it's different from the previous one
            if (sortedColors[i] != sortedColors[i - 1]) {
                deduplicatedColors.add(sortedColors[i])
            }
        }

        return deduplicatedColors
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
                            .background(Color.Black.copy(alpha = 0.5f))
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
