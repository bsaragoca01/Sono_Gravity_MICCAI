import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:web_socket_channel/io.dart'; // Import WebSocket package

void main() {
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

//localhost is common but:
//If the emulator is used: 10.0.2.2 instead of localhost

class _MyAppState extends State<MyApp> {
  late IOWebSocketChannel channel;
  late IOWebSocketChannel model_channel;
  Uint8List? imageBytes;
  Uint8List? predictedBytes;
  Uint8List? lastPredictionBytes; // Store the last prediction
  bool isFirstPrediction = true; // Track if this is the first prediction
  bool isConnected = false; // Track WebSocket connection status
  bool isPredicting = false;
  bool discardInitialFrames = true;
  int? sendTime; // Variable to hold the time when the image is sent

  @override
  void initState() {
    super.initState();

    channel = IOWebSocketChannel.connect('ws://localhost:8080');
    model_channel = IOWebSocketChannel.connect('ws://100.68.9.113:12345');

    //model_channel = IOWebSocketChannel.connect('ws://localhost:12345');

    channel.stream.listen((data) {
      if (isConnected) {
        imageBytes = Uint8List.fromList(data);
        if (!discardInitialFrames && !isPredicting) {
          predict(imageBytes!);
        }
      }
    }, onDone: ()
    {
      print("Image channel closed.");
    }, onError: (error){
      print("Error in image channel: $error");
    });

    // Listen for the prediction result from the model WebSocket
    model_channel.stream.listen((prediction) {
      final receiveTime = DateTime
          .now()
          .millisecondsSinceEpoch; // Capture receive time
      if (sendTime != null) {
        final delay = receiveTime - sendTime!; // Calculate delay
        print("Received prediction at $receiveTime ms, delay: $delay ms");
      }
      setState(() {
        lastPredictionBytes = predictedBytes;
        predictedBytes = Uint8List.fromList(prediction);
        isFirstPrediction = false;
        isPredicting = false; // Mark prediction as done
      });
    }, onDone:(){
      print("Model channel closed");
    }, onError: (error){
      print("Error in model channel: $error");
    });
    setState(() {
      isConnected = true;
      discardInitialFrames = false;
    });
  }

  void predict(Uint8List image) {
    isPredicting= true;
    sendTime = DateTime.now().millisecondsSinceEpoch; // Capture send time
    model_channel.sink.add(image);
    print("Image sent at $sendTime ms");
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text('Probe stream',
          style: TextStyle(color: Colors.white),
          ),
          backgroundColor: Color(0xFFbacae7),
        ),
        backgroundColor: Color(0xFFbacae7), // Set the background color to black
        body: Center(
          child: isFirstPrediction
              ? Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              SpinKitFadingFour(
                color: Colors.white, // Specify the color for the spinner
              ),
              SizedBox(height: 20), // Space between indicator and text
              Text('Loading',
              style: TextStyle(color:Colors.white,
                  fontWeight: FontWeight.bold
              ),
              ), // Loading message
            ],
          ) : Stack(
            alignment: Alignment.center,
            children: [
              // Display the last prediction, if available
              if (lastPredictionBytes != null)
                AnimatedOpacity(
                  opacity: (isPredicting) ? 0.0 : 1.0, // Fade out if predicting
                  duration: Duration(milliseconds: 400),
                  child: Image.memory(lastPredictionBytes!),
                ),
              // Display the new prediction only when ready
              if (predictedBytes != null)
                AnimatedOpacity(
                  opacity: (isPredicting) ? 0.0 : 1.0, // Fade out if predicting
                  duration: Duration(milliseconds: 400),
                  child: Image.memory(predictedBytes!),
                ),
              // Loading indicator
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    channel.sink.close();
    model_channel.sink.close();
    super.dispose();
  }
}

