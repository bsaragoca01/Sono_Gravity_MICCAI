import 'package:app_sono_gravity/pages/home.dart';
import 'package:app_sono_gravity/pages/loading.dart';
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
        '/loading': (context) => Loading(), //This page appears while wainting for the connection to be stablished and the predicted frames displayed.
        '/predict': (context) => MyApp(), //The predictions will be displayed in this page.
      },
    ),
  );
}
