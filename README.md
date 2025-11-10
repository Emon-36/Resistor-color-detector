**ResistorVision**

ResistorVision is an Android application built with Jetpack Compose that uses the device's camera and a machine learning model to detect and identify the color bands on a resistor.

**App PreviewFeatures** : 
	
.Live Camera Preview: Utilizes CameraX to display a real-time feed from the device's camera.
	
•AI-Powered Color Detection: Employs an ONNX (Open Neural Network Exchange) model to perform object detection on 		resistor color bands.
	
Instant Identification: Captures an image, processes it through the model, and displays the identified color sequence 		on the screen.
	
•Modern Android UI: Built entirely with Jetpack Compose and Material 3 for a clean, modern, and 				declarative user interface.
	
•Robust and Resilient: Designed to handle potential model loading failures gracefully without crashing, informing the 		user if the model fails to load.
	
	
**How It Works**

The application integrates several key Android and machine learning technologies:

1.CameraX: A Jetpack support library used to create the camera preview and capture high-resolution images for analysis.

2.ONNX Runtime (Android): The official ONNX runtime library for Android (com.microsoft.onnxruntime:onnxruntime-android:1.18.0), which loads and executes the pre-trained .onnx machine learning model directly on the device.

3.Object Detection Model: The core logic relies on an object detection model (named best.onnx) that has been trained to recognize the specific colors used on resistors.

4.Jetpack Compose: The entire user interface is built with Compose, from the camera view integration to the capture button and result text overlays.

When the "Capture" button is pressed, the app takes a picture, converts it to a bitmap, and passes it to the ObjectDetector class. This class resizes the image, creates a tensor, and feeds it to the ONNX model. The model's output, which contains bounding box and class information, is then processed to extract the color names, which are sorted and displayed to the user.

**Getting Started**

Prerequisites

•Android Studio (latest version recommended)

•An Android device with a camera

•A pre-trained ONNX model named best.onnx (see Crucial Model Requirement below)
