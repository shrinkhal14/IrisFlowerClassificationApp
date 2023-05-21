package com.example.irisflowerdetector

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import com.example.irisflowerdetector.ml.Iris

import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.w3c.dom.Text
import java.nio.ByteBuffer


class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        var button : Button = findViewById<Button>(R.id.button)
        button.setOnClickListener(View.OnClickListener {

            val ed1: TextView = findViewById(R.id.editTextNumberDecimal5)
            val ed2: TextView = findViewById(R.id.editTextNumberDecimal6)
            val ed3: TextView = findViewById(R.id.editTextNumberDecimal7)
            val ed4: TextView = findViewById(R.id.editTextNumberDecimal8)

            var v1: Float=ed1.text.toString().toFloat()
            var v2: Float=ed2.text.toString().toFloat()
            var v3: Float=ed3.text.toString().toFloat()
            var v4: Float=ed4.text.toString().toFloat()
            var byteBuffer : ByteBuffer = ByteBuffer.allocate(4*4)
            byteBuffer.putFloat(v1)
            byteBuffer.putFloat(v2)
            byteBuffer.putFloat(v3)
            byteBuffer.putFloat(v4)



            val model = Iris.newInstance(this)

// Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 4), DataType.FLOAT32)
            inputFeature0.loadBuffer(byteBuffer)

// Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

// Releases model resources if no longer used.


            var tv: TextView = findViewById(R.id.textView)
            tv.setText("Iris-Setosa-"+outputFeature0[0].toString()+"\n Iris-versicolor"+
                    outputFeature0[1].toString()+"\n Iris-Virginica"+
                    outputFeature0[2].toString())

            model.close()

        })



    }


}


