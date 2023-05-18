package org.tensorflow.lite.examples.detection;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.util.TypedValue;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.github.dhaval2404.imagepicker.ImagePicker;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.env.Utils;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    public static float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    long startTime, inferenceTime;
    int numberOfObject;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        resultText = findViewById(R.id.result);
        yolov4TinyButton = findViewById(R.id.yolov4_tiny);
        yolov5Button = findViewById(R.id.yolov5);

        yolov4TinyButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                yolov4TinyButton.setBackgroundColor(getResources().getColor(R.color.tfe_color_primary));
                yolov5Button.setBackgroundColor(getResources().getColor(R.color.tfe_color_accent));
                Toast.makeText(MainActivity.this, "Yolov4-Tiny is selected", Toast.LENGTH_SHORT).show();

                cropBitmap = Utils.processBitmap(defaultBitmap, TF_OD_API_INPUT_SIZE);
                defaultBitmap = cropBitmap.copy(cropBitmap.getConfig(), true);
                imageView.setImageBitmap(cropBitmap);
                initBox();

                try {
                    startActivity(new Intent(MainActivity.this, YOLOV4DetectorActivity.class));

                } catch (ActivityNotFoundException e) {
                    e.printStackTrace();
                    Log.e("MainActivity", "Yolov4-Tiny Activity not found", e);
                    Toast toast = Toast.makeText(getApplicationContext(), "Yolov4-Tiny Activity not found", Toast.LENGTH_SHORT);
                    toast.show();
                    finish();
                }
            }
        });

        yolov5Button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                yolov4TinyButton.setBackgroundColor(getResources().getColor(R.color.tfe_color_accent));
                yolov5Button.setBackgroundColor(getResources().getColor(R.color.tfe_color_primary));
                Toast.makeText(MainActivity.this, "Yolov5 is selected", Toast.LENGTH_SHORT).show();

                cropBitmap = Utils.processBitmap(defaultBitmap, TF_OD_API_INPUT_SIZE);
                defaultBitmap = cropBitmap.copy(cropBitmap.getConfig(),true);
                imageView.setImageBitmap(cropBitmap);
                try {
                    startActivity(new Intent(MainActivity.this, YOLOV5DetectorActivity.class));

                } catch (ActivityNotFoundException e) {
                    e.printStackTrace();
                    Log.e("MainActivity", "Yolov5 Activity not found", e);
                    Toast toast = Toast.makeText(getApplicationContext(), "Yolov5 Activity not found", Toast.LENGTH_SHORT);
                    toast.show();
                    finish();
                }
            }
        });

        this.sourceBitmap = Utils.getBitmapFromAsset(MainActivity.this, "kite.jpg");
        this.cropBitmap = Utils.processBitmap(sourceBitmap, TF_OD_API_INPUT_SIZE);
        this.defaultBitmap = cropBitmap.copy(cropBitmap.getConfig(),true);
        this.imageView.setImageBitmap(cropBitmap);

        initBox();
    }

    private Bitmap padBitmap(Bitmap Src, int padding_x, int padding_y) {
        Bitmap outputimage = Bitmap.createBitmap(Src.getWidth()+padding_x/2,Src.getHeight()+padding_y/2, Bitmap.Config.ARGB_8888);
        Canvas can1 = new Canvas(outputimage);
        can1.drawColor(Color.GRAY);
        can1.drawBitmap(Src, padding_x/2, padding_y/2, null);
        Bitmap output = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE,TF_OD_API_INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Canvas can2 = new Canvas(output);
        can2.drawColor(Color.GRAY);
        can2.drawBitmap(outputimage, 0, 0, null);
        return output;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == Activity.RESULT_OK) {
            Uri uri = data.getData();

            try {
                this.sourceBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                e.printStackTrace();
            }
            this.cropBitmap = Utils.processBitmap(sourceBitmap, TF_OD_API_INPUT_SIZE);
            this.defaultBitmap = cropBitmap.copy(cropBitmap.getConfig(),true);
            this.imageView.setImageBitmap(cropBitmap);
            initBox();
        } else if (resultCode == ImagePicker.RESULT_ERROR) {
            Toast.makeText(this, ImagePicker.getError(data), Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(this, "Task Cancelled", Toast.LENGTH_SHORT).show();
        }

    }

    private static final Logger LOGGER = new Logger();
    public static int TF_OD_API_INPUT_SIZE = 1024;

    // Minimum detection confidence to track a detection.
    private static final boolean MAINTAIN_ASPECT = false;
    private Integer sensorOrientation = 90;

    private Classifier detector;
    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;
    private MultiBoxTracker tracker;
    private OverlayView trackingOverlay;
    protected int previewWidth = 0;
    protected int previewHeight = 0;
    private Bitmap sourceBitmap;
    private Bitmap cropBitmap, defaultBitmap;

    private Button yolov4TinyButton, yolov5Button;
    private ImageView imageView;
    private TextView resultText;


    private void initBox() {
        previewHeight = TF_OD_API_INPUT_SIZE;
        previewWidth = TF_OD_API_INPUT_SIZE;
        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        tracker = new MultiBoxTracker(this);
        trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                canvas -> tracker.draw(canvas));

        tracker.setFrameConfiguration(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, sensorOrientation);
    }

    private void handleResult(Bitmap bitmap, List<Classifier.Recognition> results) {
        final Canvas canvas = new Canvas(bitmap);
        final Paint paint = new Paint();
        final Paint fgPaint;
        final float textSizePx;
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2.0f);
        numberOfObject = 0;
        textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, 8, getResources().getDisplayMetrics());
        fgPaint = new Paint();
        fgPaint.setTextSize(textSizePx);
        fgPaint.setColor(Color.RED);

        final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();
        for (final Classifier.Recognition result : results) {
            final RectF location = result.getLocation();
            if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
                numberOfObject++;
                canvas.drawText(String.format("%.2f", result.getConfidence()), (int) location.left+5, location.top+20, fgPaint);
                canvas.drawRect(location, paint);
            }
        }
        resultText.setText("Objects: " + String.valueOf(numberOfObject) + "\nInference time: "+String.valueOf(inferenceTime)+"ms");

        imageView.setImageBitmap(bitmap);
    }
}
