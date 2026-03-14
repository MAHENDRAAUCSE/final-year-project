import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = "http://10.0.2.2:8000";
  static const String webBaseUrl = "http://localhost:8000";

  // For mobile (Android/iOS) - uses file path
  static Future uploadFile(String filePath) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse("$baseUrl/predict"),
      );

      var file = File(filePath);
      var stream = http.ByteStream(file.openRead());
      var length = await file.length();

      var multipartFile = http.MultipartFile(
        'file',
        stream,
        length,
        filename: file.path.split('/').last,
      );

      request.files.add(multipartFile);
      var response = await request.send();
      var respStr = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        return jsonDecode(respStr);
      } else {
        throw Exception("Upload failed: ${response.statusCode}");
      }
    } catch (e) {
      rethrow;
    }
  }

  // For web - uses bytes
  static Future uploadFileWeb(String fileName, List<int> fileBytes) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse("$webBaseUrl/predict"),
      );

      var multipartFile = http.MultipartFile(
        'file',
        http.ByteStream.fromBytes(fileBytes),
        fileBytes.length,
        filename: fileName,
      );

      request.files.add(multipartFile);
      var response = await request.send();
      var respStr = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        return jsonDecode(respStr);
      } else {
        throw Exception("Upload failed: ${response.statusCode}");
      }
    } catch (e) {
      rethrow;
    }
  }
}