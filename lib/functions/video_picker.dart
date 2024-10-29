import 'package:file_picker/file_picker.dart';
import 'dart:io';


class VideoPicker {
  Future<File?> pickVideo() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.video,
      );

      if (result != null) {
        File file = File(result.files.single.path!);
        // Simulate loading process
        await Future.delayed(Duration(seconds: 2));
        return file;
      } else {
        return null;
      }
    } catch (e) {
      print('Error picking video: $e');
      return null;
    }
  }
}
