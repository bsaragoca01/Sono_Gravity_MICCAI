import 'package:app_sono_gravity/pages/home.dart';
import 'package:app_sono_gravity/pages/choose_location.dart';
import 'package:app_sono_gravity/pages/loading.dart';
import 'package:app_sono_gravity/pages/predictions.dart';
import 'package:flutter/material.dart';
//import 'package:app_sono_gravity/pages/test.dart';

void main() {
  // Ensures that all bindings are initialized properly before running the app
  //WidgetsFlutterBinding.ensureInitialized();

  // Initialize the ONNX Runtime environment
  //OrtEnv.instance.init();  // If init() is synchronous, no await needed

  // Run the Flutter app
  runApp(
    MaterialApp(

      initialRoute: '/home',
      routes: {
        '/': (context) => Home(), // '/' is the home page
        '/home': (context) => Home(),
        '/loading': (context) => Loading(),
        //'/choose_location': (context) => const ChooseLocation(),
        '/predict': (context) => MyApp(),
      },
    ),
  );
}


