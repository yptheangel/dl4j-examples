/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.examples.image_recognition_demo;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.os.Build;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import android.os.AsyncTask;
import android.widget.TextView;
import android.widget.Toast;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.examples.image_recognition_demo.datasetHelper.MnistDataSetIteratorAndroid;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.text.DecimalFormat;
import java.util.concurrent.TimeUnit;

public class MainActivity extends AppCompatActivity {

    MainActivity.DrawingView drawingView;
    String absolutePath;
    public static INDArray output;
    // for permission requests
    public static final int REQUEST_PERMISSION = 123;


    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        RelativeLayout parent = findViewById(R.id.layout2);
        drawingView = new MainActivity.DrawingView(this);
        parent.addView(drawingView);

        // request permission to write data (aka images) to the user's external storage of their phone
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
            && ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                REQUEST_PERMISSION);
        }

        // request permission to read data (aka images) from the user's external storage of their phone
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
            && ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED)
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                REQUEST_PERMISSION);

        //Configure what does the Train button do
        Button train_button = findViewById(R.id.train_button);
        train_button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                AsyncTaskTrainer trainer = new AsyncTaskTrainer();
                trainer.execute(absolutePath);
                Context context = getApplicationContext();
                Toast.makeText(context, "Model training will now start", Toast.LENGTH_SHORT).show();
                onProgressBar();
            }
        });

        //Configure what does the Predict button do
        Button predict_button = findViewById(R.id.pred_button);
        predict_button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
//                absolutePath = saveDrawing();
                loadImageFromStorage(absolutePath);
                onProgressBar();
                AsyncTaskPredictor runner = new AsyncTaskPredictor();
                runner.execute(absolutePath);
            }
        });

    }

    private class AsyncTaskPredictor extends AsyncTask<String, Integer, INDArray> {

        // Runs in UI before background thread is called
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
        }

        @Override
        protected INDArray doInBackground(String... params) {
            // Main background thread, this will load the model and test the input image
            int height = 28;
            int width = 28;
            int channels = 1;

            //load the model from the raw folder with a try / catch block
            try {
                // Load the pretrained network.
                String modelFilename = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath() + "/mnist_classifier.zip";
                MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFilename);
                Log.d("Debug", "Model is loaded from "+modelFilename);

                //load the image file to test
                File f=new File(absolutePath, "drawn_image.jpg");

                //Use the nativeImageLoader to convert to numerical matrix
                NativeImageLoader loader = new NativeImageLoader(height, width, channels);

//                put image into INDArray
                INDArray image = loader.asMatrix(f);

                //pass through neural net and store it in output array
                output = model.output(image);

            } catch (IOException e) {
                e.printStackTrace();
            }
            return output;
        }

        @Override
        protected void onProgressUpdate(Integer... values) {
            super.onProgressUpdate(values);
        }


        @Override
        protected void onPostExecute(INDArray result) {
            super.onPostExecute(result);

            //used to control the number of decimals places for the output probability
            DecimalFormat df2 = new DecimalFormat(".##");

            //transfer the neural network output to an array
            double[] results = {result.getDouble(0,0),result.getDouble(0,1),result.getDouble(0,2),
                result.getDouble(0,3),result.getDouble(0,4),result.getDouble(0,5),result.getDouble(0,6),
                result.getDouble(0,7),result.getDouble(0,8),result.getDouble(0,9),};

            //find the UI tvs to display the prediction and confidence values
            TextView out1 = findViewById(R.id.prediction);
            TextView out2 = findViewById(R.id.confidence);

            //display the values using helper functions defined below
            out2.setText(String.valueOf(df2.format(arrayMaximum(results))));
            out1.setText(String.valueOf(getIndexOfLargestValue(results)));

            //helper function to turn off progress test
            offProgressBar();
        }

    }

    private class AsyncTaskTrainer extends AsyncTask<String, Integer, INDArray>
    {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
        }

        @Override
        protected INDArray doInBackground(String... params) {

            int height = 28;
            int width = 28;
            int channels = 1; // single channel for grayscale images
            int outputNum = 10; // 10 digits classification
            int batchSize = 100;
            int nEpochs = 1;
            double learningRate = 0.001;
            MultiLayerNetwork model = null;

            int seed = 1234;
            String modelFilename = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath() + "/mnist_classifier.zip";
            Log.d("Debug","Model will be saved here: "+modelFilename);

            try {
                Log.d("Debug","Data load and vectorization...");
                DataSetIterator mnistTrain = new MnistDataSetIteratorAndroid(batchSize,true, seed);
                DataSetIterator mnistTest = new MnistDataSetIteratorAndroid(batchSize,false, seed);
                Log.d("Debug","Network configuration and training...");

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .updater(new Nesterovs(learningRate, Nesterovs.DEFAULT_NESTEROV_MOMENTUM))
                    .weightInit(WeightInit.XAVIER)
                    .list()
                    .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                    .layer(2, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(50).build())
                    .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                    .setInputType(InputType.convolutionalFlat(height, width, 1)) // InputType.convolutional for normal image
                    .backpropType(BackpropType.Standard)
                    .build();

                model = new MultiLayerNetwork(conf);
                model.init();
                Log.d("Debug","\n"+model.summary());

                model.setListeners(new ScoreIterationListener());
                long start_time=System.nanoTime();
                for (int i =1; i < nEpochs+1; i++) {
                    model.fit(mnistTrain);

                    Log.d("Debug","Completed epoch "+ i);
                    mnistTrain.reset();
                }
                long elapsed_time= System.nanoTime()-start_time;
                long elapsed_time_seconds= TimeUnit.SECONDS.convert(elapsed_time,TimeUnit.NANOSECONDS);
                Log.d("Debug","Model Training took "+elapsed_time_seconds+" seconds");

                ModelSerializer.writeModel(model, modelFilename, true);
                Log.d("Debug","Model Training is Done.");

                Evaluation eval = model.evaluate(mnistTest);
                Log.d("Debug",eval.stats());
                mnistTest.reset();

            } catch (IOException e) {
                e.printStackTrace();
            }
            offProgressBar();
            return output;
        }
    }

    //code for the drawing input
    public class DrawingView extends View {

        private Path    mPath;
        private Paint   mBitmapPaint;
        private Paint   mPaint;
        private Bitmap  mBitmap;
        private Canvas  mCanvas;

        public DrawingView(Context c) {
            super(c);

            mPath = new Path();
            mBitmapPaint = new Paint(Paint.DITHER_FLAG);
            mPaint = new Paint();
            mPaint.setAntiAlias(true);
            mPaint.setStrokeJoin(Paint.Join.ROUND);
            mPaint.setStrokeCap(Paint.Cap.ROUND);
            mPaint.setStrokeWidth(60);
            mPaint.setDither(true);
            mPaint.setColor(Color.MAGENTA);
            mPaint.setStyle(Paint.Style.STROKE);
        }

        @Override
        protected void onSizeChanged(int W, int H, int oldW, int oldH) {
            super.onSizeChanged(W, H, oldW, oldH);
            mBitmap = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_4444);
            mCanvas = new Canvas(mBitmap);
        }

        @Override
        protected void onDraw(Canvas canvas) {
            canvas.drawBitmap(mBitmap, 0, 0, mBitmapPaint);
            canvas.drawPath(mPath, mPaint);
        }

        private float mX, mY;
        private static final float TOUCH_TOLERANCE = 4;

        private void touch_start(float x, float y) {
            mPath.reset();
            mPath.moveTo(x, y);
            mX = x;
            mY = y;
        }
        private void touch_move(float x, float y) {
            float dx = Math.abs(x - mX);
            float dy = Math.abs(y - mY);
            if (dx >= TOUCH_TOLERANCE || dy >= TOUCH_TOLERANCE) {
                mPath.quadTo(mX, mY, (x + mX)/2, (y + mY)/2);
                mX = x;
                mY = y;
            }
        }
        private void touch_up() {
            mPath.lineTo(mX, mY);
            mCanvas.drawPath(mPath, mPaint);
            mPath.reset();
        }

        @Override
        public boolean onTouchEvent(MotionEvent event) {
            float x = event.getX();
            float y = event.getY();

            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    invalidate();
                    clear();
                    touch_start(x, y);
                    invalidate();
                    break;
                case MotionEvent.ACTION_MOVE:
                    touch_move(x, y);
                    invalidate();
                    break;
                case MotionEvent.ACTION_UP:
                    touch_up();
                    absolutePath = saveDrawing();
                    invalidate();
                    clear();
                    break;
            }
            return true;
        }

        public void clear(){
            mBitmap.eraseColor(Color.TRANSPARENT);
            invalidate();
            System.gc();
        }
    }

    public String  saveDrawing(){
        drawingView.destroyDrawingCache();
        drawingView.setDrawingCacheEnabled(true);
        Bitmap b = drawingView.getDrawingCache();

        File directory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
        File mypath=new File(directory,"drawn_image.jpg");

        //use a fileOutputStream to write the file to the location in a try / catch block
        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream(mypath);
            b.compress(Bitmap.CompressFormat.JPEG, 100, fos);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                fos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return directory.getAbsolutePath();
    }

    private void loadImageFromStorage(String path)
    {

        //use a fileInputStream to read the file in a try / catch block
        try {
            File f=new File(path, "drawn_image.jpg");
            Bitmap b = BitmapFactory.decodeStream(new FileInputStream(f));
            ImageView img=findViewById(R.id.outputView);
            img.setImageBitmap(b);
        }
        catch (FileNotFoundException e)
        {
            e.printStackTrace();
        }
    }

    public void onProgressBar(){
        TextView bar = findViewById(R.id.processing);
        bar.setVisibility(View.VISIBLE);
    }

    public void offProgressBar(){
        TextView bar = findViewById(R.id.processing);
        bar.setVisibility(View.INVISIBLE);
    }

    //helper class to return the largest value in the output array
    public static double arrayMaximum(double[] arr) {
        double max = Double.NEGATIVE_INFINITY;
        for(double cur: arr)
            max = Math.max(max, cur);
        return max;
    }

    // helper class to find the index (and therefore numerical value) of the largest confidence score
    public int getIndexOfLargestValue( double[] array )
    {
        if ( array == null || array.length == 0 ) return -1;
        int largest = 0;
        for ( int i = 1; i < array.length; i++ )
        {if ( array[i] > array[largest] ) largest = i;            }
        return largest;
    }

}





