import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:web_socket_channel/io.dart'; 

void main() {
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  late IOWebSocketChannel channel;
  late IOWebSocketChannel model_channel;
  Uint8List? imageBytes;
  Uint8List? predictedBytes;
  Uint8List? lastPredictionBytes; 
  bool isFirstPrediction = true; 
  bool isConnected = false;
  bool isPredicting = false;
  bool discardInitialFrames = true;
  int? sendTime;

  @override
  void initState() {
    super.initState();
    channel = IOWebSocketChannel.connect('ws://path_to_the_server'); //server which is displaying the Ultrasound frames in real-time
    model_channel = IOWebSocketChannel.connect('ws://path_to_server_predictions'); //server which is responsible to apply the DL models trained previously
    channel.stream.listen((data) {
      if (isConnected) {
        imageBytes = Uint8List.fromList(data);
        if (!discardInitialFrames && !isPredicting) { //The frames before the Websocket handshake are discarded, so they are not accumulated turning the approach not in real-time 
          predict(imageBytes!);
        }
      }
    }, onDone: ()
    {
      print("Image channel closed.");
    }, onError: (error){
      print("Error in image channel: $error");
    });

    //Listen for the prediction result 
    model_channel.stream.listen((prediction) {
      final receiveTime = DateTime
          .now()
          .millisecondsSinceEpoch; //Capture the receive time
      if (sendTime != null) {
        final delay = receiveTime - sendTime!; //Calculate delay, the time between sending the frame and getting its prediction back
        print("Received prediction at $receiveTime ms, delay: $delay ms");
      }
      setState(() {
        lastPredictionBytes = predictedBytes;
        predictedBytes = Uint8List.fromList(prediction);
        isFirstPrediction = false;
        isPredicting = false; //Mark prediction as done
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
    sendTime = DateTime.now().millisecondsSinceEpoch; //Capture send time
    model_channel.sink.add(image);
    print("Image sent at $sendTime ms");
  }

  @override
  Widget build(BuildContext context) { //Page design
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text('Probe stream',
          style: TextStyle(color: Colors.white),
          ),
          backgroundColor: Color(0xFFbacae7),
        ),
        backgroundColor: Color(0xFFbacae7), 
        body: Center(
          child: isFirstPrediction
              ? Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              SpinKitFadingFour(
                color: Colors.white,
              ),
              SizedBox(height: 20),
              Text('Loading',
              style: TextStyle(color:Colors.white,
                  fontWeight: FontWeight.bold
              ),
              ), // Loading message
            ],
          ) : Stack(
            alignment: Alignment.center,
            children: [
              //Display the previous prediction while the new one is being processed to avoid visual cuts between frames
              //Smoother transition but does not affect the time between changing predictions.
              if (lastPredictionBytes != null)
                AnimatedOpacity(
                  opacity: (isPredicting) ? 0.0 : 1.0,
                  duration: Duration(milliseconds: 400),
                  child: Image.memory(lastPredictionBytes!),
                ),
              //Display the new prediction only when ready
              if (predictedBytes != null)
                AnimatedOpacity(
                  opacity: (isPredicting) ? 0.0 : 1.0, 
                  duration: Duration(milliseconds: 400),
                  child: Image.memory(predictedBytes!),
                ),
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
