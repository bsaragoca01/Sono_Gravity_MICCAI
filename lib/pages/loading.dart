import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:onnxruntime/onnxruntime.dart'; // Import ONNX Runtime
import 'package:http/http.dart' as http;

class Loading extends StatefulWidget {
  const Loading({super.key});

  @override
  _LoadingState createState() => _LoadingState();
}



class _LoadingState extends State<Loading> {
  String _loadingStatus = "Loading model...";

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  // Future<void> _loadModel() async {
  //   try {
  //     // Initialize session options
  //     //final sessionOptions = OrtSessionOptions();
  //
  //     // Load model from assets
  //     final startTime = DateTime.now();
  //     final sessionOptions = OrtSessionOptions();
  //     const assetFileName = 'assets/model_experiment1.onnx';
  //     final rawAssetFile = await rootBundle.load(assetFileName);
  //     final bytes = rawAssetFile.buffer.asUint8List();
  //     final session = OrtSession.fromBuffer(bytes, sessionOptions);
  //     final endTime = DateTime.now();
  //     print('Model loaded in ${endTime.difference(startTime).inMilliseconds} ms');
  //
  //     // Create a session from the loaded model
  //     //final session = OrtSession.fromBuffer(bytes, sessionOptions);
  //
  //     // Navigate to the next screen with the session
  //     Navigator.pushReplacementNamed(context, '/predict', arguments: session);
  //   } catch (e) {
  //     // Handle exceptions
  //     print('Failed to load the model: $e');
  //     // Optionally, show an error message to the user
  //     ScaffoldMessenger.of(context).showSnackBar(
  //       SnackBar(
  //         content: Text('Failed to load the model: $e'),
  //         backgroundColor: Colors.red,
  //       ),
  //     );
  //   }
  // }

  Future<void> _loadModel() async {
    print("Send GET request");
    try {
      final response = await http.get(Uri.parse('http://localhost:5000/load_model'));  // Flask server for loading model
      if (response.statusCode == 200) {
        print("successful");
        setState(() {
          _loadingStatus = "Model loaded successfully!";
        });
        // Navigate to the prediction page after model is loaded
        Navigator.pushReplacementNamed(context, '/predict');
      } else {
        setState(() {
          print("merda");
          _loadingStatus = "Failed to load the model.";
        });
      }
    } catch (e) {
      setState(() {
        _loadingStatus = "Error: $e";
        print(e);
      });
    }
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: SpinKitFadingFour(
          color: Colors.white,
          size: 100.0,
        ),
      ),
    );
  }
}