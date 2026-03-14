import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import '../services/api_service.dart';
import 'prediction_screen.dart';

class UploadScreen extends StatefulWidget {
  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  bool isLoading = false;
  String? selectedFileName;
  String? errorMessage;

  void uploadDataset() async {
    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['csv'],
        withData: true,
      );

      if (result != null) {
        setState(() {
          selectedFileName = result.files.single.name;
          isLoading = true;
          errorMessage = null;
        });

        var response = await ApiService.uploadFileWeb(
          result.files.single.name,
          result.files.single.bytes!,
        );
        
        setState(() {
          isLoading = false;
        });

        if (response != null && response['future_predictions'] != null) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => PredictionScreen(
                predictions: response['future_predictions'],
              ),
            ),
          );
        } else {
          setState(() {
            errorMessage = "Invalid response from server";
          });
        }
      }
    } catch (e) {
      setState(() {
        isLoading = false;
        errorMessage = "Error: ${e.toString()}";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Upload Dataset")),
      body: Center(
        child: Padding(
          padding: EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (selectedFileName != null)
                Padding(
                  padding: EdgeInsets.only(bottom: 16.0),
                  child: Text(
                    "Selected: $selectedFileName",
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                  ),
                ),
              ElevatedButton.icon(
                icon: Icon(Icons.upload_file),
                label: Text(isLoading ? "Uploading..." : "Upload CSV"),
                onPressed: isLoading ? null : uploadDataset,
              ),
              if (isLoading)
                Padding(
                  padding: EdgeInsets.only(top: 16.0),
                  child: CircularProgressIndicator(),
                ),
              if (errorMessage != null)
                Padding(
                  padding: EdgeInsets.only(top: 16.0),
                  child: Text(
                    errorMessage!,
                    style: TextStyle(color: Colors.red, fontSize: 14),
                    textAlign: TextAlign.center,
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}