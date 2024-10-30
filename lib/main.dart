import 'package:app_sono_gravity/pages/home.dart';
import 'package:app_sono_gravity/pages/predictions.dart';
import 'package:flutter/material.dart';

void main() {
  //Run the Flutter app
  runApp(
    MaterialApp(
      initialRoute: '/home', 
      routes: {
        '/': (context) => Home(), //'/' is the home page. The app initiates in this page.
        '/home': (context) => Home(),
        '/predict': (context) => MyApp(), //The predictions will be displayed in this page.
      },
    ),
  );
}
