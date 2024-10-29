import 'package:flutter/material.dart';
import 'dart:io';
import 'package:app_sono_gravity/functions/video_picker.dart';
import 'package:onnxruntime/onnxruntime.dart';


class ChooseLocation extends StatefulWidget {
  const ChooseLocation({super.key});

  @override
  _ChooseLocationState createState() => _ChooseLocationState();
}

class _ChooseLocationState extends State<ChooseLocation> {
  //late Model _model;
  String? _fileName;
  String? _filePath;
  bool _isLoading = false;
  VideoPicker _videoPicker = VideoPicker();

  late OrtSession _model; // Define a variable to hold the model

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _model = ModalRoute.of(context)!.settings.arguments as OrtSession; // Receive the model
  }


  Future<void> _pickVideo() async {
    setState(() {
      _isLoading = true;
    });

    File? pickedFile = await _videoPicker.pickVideo();

    if (pickedFile != null) {
      setState(() {
        _fileName = pickedFile.path.split('/').last;
        _filePath = pickedFile.path;
        _isLoading = false;
      });
    } else {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _clearSelection() {
    setState(() {
      _fileName = null;
      _filePath = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    //final model = ModalRoute.of(context)!.settings.arguments as Model;
    return Scaffold(
      backgroundColor: Colors.grey[100],
      body: SafeArea(
        child: Container(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(0, 500.0, 0, 0),
            child: Column(
              children: [
                SizedBox(height: 20.0),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Card(
                      color: Colors.white,
                      elevation: 8.0,
                      child: Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Column(
                          children: [
                            Text(
                              'Select a video from your device',
                              style: TextStyle(
                                fontSize: 25.0,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            SizedBox(height: 16),
                            _isLoading
                                ? CircularProgressIndicator()
                                : _fileName != null
                                ?
                            Row(
                              children: [
                                Text(
                                  _fileName!,
                                  style: TextStyle(fontSize: 20.0),
                                ),
                                IconButton(
                                  icon: Icon(Icons.clear),
                                  onPressed: _clearSelection,
                                ),
                                SizedBox(height: 16),
                                //SizedBox(height: 16),
                                ElevatedButton(
                                  onPressed: (){
                                    Navigator.pushNamed(
                                      context,
                                      '/predict',
                                      arguments: {
                                        'filePath': _filePath!,
                                        //'model': model,
                                      },
                                    );
                                  },
                                  child: Text(
                                    'Done',
                                    style: TextStyle(
                                      color: Colors.white,
                                      fontWeight: FontWeight.bold,
                                      fontSize: 20.0,
                                    ),
                                  ),
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: Colors.purple,
                                    //padding: EdgeInsets.symmetric(horizontal: 50, vertical: 20),
                                    //textStyle: TextStyle(
                                    //fontSize: 30,
                                    //fontWeight: FontWeight.bold))
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(3),
                                    ),
                                    padding: EdgeInsets.all(5),
                                  ),
                                ),
                              ],
                            )
                                : ElevatedButton.icon(
                              onPressed: _pickVideo, // Set the function here
                              icon: Icon(
                                Icons.drive_folder_upload,
                                color: Colors.grey[500],
                                size: 32.0,
                              ),
                              label: Text(
                                'Choose a file',
                                style: TextStyle(
                                  fontSize: 22.0,
                                ),
                              ),
                              style: ElevatedButton.styleFrom(
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(0),
                                ),
                                padding: EdgeInsets.all(5),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
