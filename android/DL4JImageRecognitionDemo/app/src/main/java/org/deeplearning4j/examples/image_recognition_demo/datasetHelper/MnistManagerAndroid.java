package org.deeplearning4j.examples.image_recognition_demo.datasetHelper;



import org.deeplearning4j.datasets.mnist.MnistImageFile;
import org.deeplearning4j.datasets.mnist.MnistLabelFile;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class MnistManagerAndroid {
    MnistImageFile images;
    private MnistLabelFile labels;

    private byte[][] imagesArr;
    private int[] labelsArr;
    private static final int HEADER_SIZE = 8;

    /**
     * Writes the given image in the given file using the PPM data format.
     *
     * @param image
     * @param ppmFileName
     * @throws IOException
     */
    public static void writeImageToPpm(int[][] image, String ppmFileName) throws IOException {
        try (BufferedWriter ppmOut = new BufferedWriter(new FileWriter(ppmFileName))) {
            int rows = image.length;
            int cols = image[0].length;
            ppmOut.write("P3\n");
            ppmOut.write("" + rows + " " + cols + " 255\n");
            for (int i = 0; i < rows; i++) {
                StringBuilder s = new StringBuilder();
                for (int j = 0; j < cols; j++) {
                    s.append(image[i][j] + " " + image[i][j] + " " + image[i][j] + "  ");
                }
                ppmOut.write(s.toString());
            }
        }

    }

    /**
     * Constructs an instance managing the two given data files. Supports
     * <code>NULL</code> value for one of the arguments in case reading only one
     * of the files (images and labels) is required.
     *
     * @param imagesFile
     *            Can be <code>NULL</code>. In that case all future operations
     *            using that file will fail.
     * @param labelsFile
     *            Can be <code>NULL</code>. In that case all future operations
     *            using that file will fail.
     * @throws IOException
     */
    public MnistManagerAndroid(String imagesFile, String labelsFile, boolean train) throws IOException {
        this(imagesFile, labelsFile, train ? MnistDataFetcherAndroid.NUM_EXAMPLES : MnistDataFetcherAndroid.NUM_EXAMPLES_TEST);
    }

    public MnistManagerAndroid(String imagesFile, String labelsFile, int numExamples) throws IOException {
        if (imagesFile != null) {
            images = new MnistImageFile(imagesFile, "r");
            imagesArr = images.readImagesUnsafe(numExamples);
        }
        if (labelsFile != null) {
            labels = new MnistLabelFile(labelsFile, "r");
            labelsArr = labels.readLabels(numExamples);
        }
    }

    public MnistManagerAndroid(String imagesFile, String labelsFile) throws IOException {
        this(imagesFile, labelsFile, true);
    }

    /**
     * Reads the current image.
     *
     * @return matrix
     * @throws IOException
     */
    public int[][] readImage() throws IOException {
        if (images == null) {
            throw new IllegalStateException("Images file not initialized.");
        }
        return images.readImage();
    }

    public byte[] readImageUnsafe(int i) {
        return imagesArr[i];
    }

    /**
     * Set the position to be read.
     *
     * @param index
     */
    public void setCurrent(int index) {
        images.setCurrentIndex(index);
        labels.setCurrentIndex(index);
    }

    /**
     * Reads the current label.
     *
     * @return int
     * @throws IOException
     */
    public int readLabel() throws IOException {
        if (labels == null) {
            throw new IllegalStateException("labels file not initialized.");
        }
        return labels.readLabel();
    }

    public int readLabel(int i) {
        return labelsArr[i];
    }

    /**
     * Get the underlying images file as {@link MnistImageFile}.
     *
     * @return {@link MnistImageFile}.
     */
    public MnistImageFile getImages() {
        return images;
    }

    /**
     * Get the underlying labels file as {@link MnistLabelFile}.
     *
     * @return {@link MnistLabelFile}.
     */
    public MnistLabelFile getLabels() {
        return labels;
    }

    /**
     * Close any resources opened by the manager.
     */
    public void close() {
        if (images != null) {
            try {
                images.close();
            } catch (IOException e) {
            }
            images = null;
        }
        if (labels != null) {
            try {
                labels.close();
            } catch (IOException e) {
            }
            labels = null;
        }
    }
}
