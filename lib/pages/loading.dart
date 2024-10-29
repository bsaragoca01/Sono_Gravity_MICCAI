import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:http/http.dart' as http;

class Loading extends StatefulWidget {
  const Loading({super.key});
  @override
  _LoadingState createState() => _LoadingState();
}

class _LoadingState extends State<Loading> {
  //Loading message displayed initially
  String _loadingStatus = "Loading model...";
  @override
  void initState() {
    super.initState();
    _loadModel();
  }
  Future<void> _loadModel() async {
    //A GET request is made to load the model
    print("Send GET request");
    try {
      final response = await http.get(Uri.parse('http://localhost:5000/load_model'));
      //It is verified if the request to load the model is successful (200)
      if (response.statusCode == 200) {
        print("successful");
        setState(() {
          _loadingStatus = "Model loaded successfully!";
        });
        // Once it is successful, the page is directed to the Prediction page
        Navigator.pushReplacementNamed(context, '/predict');
      } else {
        setState(() {
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
  Widget build(BuildContext context) { //"Loading" page design
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
