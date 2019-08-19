package org.deeplearning4j.examples.image_recognition_demo.datasetHelper;

import android.os.Environment;
import android.util.Log;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.MathUtils;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.zip.Adler32;
import java.util.zip.Checksum;

/**
 * Data fetcher for the MNIST dataset
 * @author Adam Gibson
 * @modifications for Android Wilson Choo
 */

public class MnistDataFetcherAndroid extends BaseDataFetcher{
    public static final int NUM_EXAMPLES = 60000;
    public static final int NUM_EXAMPLES_TEST = 10000;

    protected static final long CHECKSUM_TRAIN_FEATURES = 2094436111L;
    protected static final long CHECKSUM_TRAIN_LABELS = 4008842612L;
    protected static final long CHECKSUM_TEST_FEATURES = 2165396896L;
    protected static final long CHECKSUM_TEST_LABELS = 2212998611L;

    protected static final long[] CHECKSUMS_TRAIN = new long[]{CHECKSUM_TRAIN_FEATURES, CHECKSUM_TRAIN_LABELS};
    protected static final long[] CHECKSUMS_TEST = new long[]{CHECKSUM_TEST_FEATURES, CHECKSUM_TEST_LABELS};

//    protected transient MnistManager man;
    protected transient MnistManagerAndroid man;

    protected boolean binarize = true;
    protected boolean train;
    protected int[] order;
    protected Random rng;
    protected boolean shuffle;
    protected boolean oneIndexed = false;
    protected boolean fOrder = false; //MNIST is C order, EMNIST is F order

    protected boolean firstShuffle = true;
    protected final int numExamples;


    /**
     * Constructor telling whether to binarize the dataset or not
     * @param binarize whether to binarize the dataset or not
     * @throws IOException
     */
    public MnistDataFetcherAndroid(boolean binarize) throws IOException {
        this(binarize, true, true, System.currentTimeMillis(), NUM_EXAMPLES);
    }

    public MnistDataFetcherAndroid(boolean binarize, boolean train, boolean shuffle, long rngSeed, int numExamples) throws IOException {
//        if (!mnistExists()) {
//            new MnistFetcherAndroid().downloadAndUntar();
//        }

//        String MNIST_ROOT = DL4JResources.getDirectory(ResourceType.DATASET, "MNIST").getAbsolutePath();
        String MNIST_ROOT= Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath();

//        String MNIST_ROOT = new ClassPathResource("raw").getFile().toString();

//        String MNIST_ROOT=R.raw.mnist_classifier;
//        InputStream inputStream = getResources().openRawResource(R.raw.mnist_classifier);

        Log.d("Debug","Mnist Dataset is saved at : "+MNIST_ROOT);
        String images;
        String labels;
        long[] checksums;
        if (train) {
            images = FilenameUtils.concat(MNIST_ROOT, MnistFetcherAndroid.TRAINING_FILES_FILENAME_UNZIPPED);
            labels = FilenameUtils.concat(MNIST_ROOT, MnistFetcherAndroid.TRAINING_FILE_LABELS_FILENAME_UNZIPPED);
            totalExamples = NUM_EXAMPLES;
            checksums = CHECKSUMS_TRAIN;
        } else {
            images = FilenameUtils.concat(MNIST_ROOT, MnistFetcherAndroid.TEST_FILES_FILENAME_UNZIPPED);
            labels = FilenameUtils.concat(MNIST_ROOT, MnistFetcherAndroid.TEST_FILE_LABELS_FILENAME_UNZIPPED);
            totalExamples = NUM_EXAMPLES_TEST;
            checksums = CHECKSUMS_TEST;
        }
        String[] files = new String[]{images, labels};

        try {
            man = new MnistManagerAndroid(images, labels, train);
            validateFiles(files, checksums);
        } catch (Exception e) {
            try {
                FileUtils.deleteDirectory(new File(MNIST_ROOT));
            } catch (Exception e2){ }
            new MnistFetcherAndroid().downloadAndUntar();
            man = new MnistManagerAndroid(images, labels, train);
            validateFiles(files, checksums);
        }

        numOutcomes = 10;
        this.binarize = binarize;
        cursor = 0;
        inputColumns = man.getImages().getEntryLength();
        this.train = train;
        this.shuffle = shuffle;

        if (train) {
            order = new int[NUM_EXAMPLES];
        } else {
            order = new int[NUM_EXAMPLES_TEST];
        }
        for (int i = 0; i < order.length; i++)
            order[i] = i;
        rng = new Random(rngSeed);
        this.numExamples = numExamples;
        reset(); //Shuffle order
    }

    private boolean mnistExists() {
        String MNIST_ROOT = DL4JResources.getDirectory(ResourceType.DATASET, "MNIST").getAbsolutePath();
        //Check 4 files:
        File f = new File(MNIST_ROOT, MnistFetcherAndroid.TRAINING_FILES_FILENAME_UNZIPPED);
        if (!f.exists())
            return false;
        f = new File(MNIST_ROOT, MnistFetcherAndroid.TRAINING_FILE_LABELS_FILENAME_UNZIPPED);
        if (!f.exists())
            return false;
        f = new File(MNIST_ROOT, MnistFetcherAndroid.TEST_FILES_FILENAME_UNZIPPED);
        if (!f.exists())
            return false;
        f = new File(MNIST_ROOT, MnistFetcherAndroid.TEST_FILE_LABELS_FILENAME_UNZIPPED);
        if (!f.exists())
            return false;
        return true;
    }

    private void validateFiles(String[] files, long[] checksums){
        //Validate files:
        try {
            for (int i = 0; i < files.length; i++) {
                File f = new File(files[i]);
                Checksum adler = new Adler32();
                long checksum = f.exists() ? FileUtils.checksum(f, adler).getValue() : -1;
                if (!f.exists() || checksum != checksums[i]) {
                    throw new IllegalStateException("Failed checksum: expected " + checksums[i] +
                            ", got " + checksum + " for file: " + f);
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public MnistDataFetcherAndroid() throws IOException {
        this(true);
    }

    @Override
    public void fetch(int numExamples) {
        if (!hasMore()) {
            throw new IllegalStateException("Unable to get more; there are no more images");
        }

        float[][] featureData = new float[numExamples][0];
        float[][] labelData = new float[numExamples][0];

        int actualExamples = 0;
        byte[] working = null;
        for (int i = 0; i < numExamples; i++, cursor++) {
            if (!hasMore())
                break;

            byte[] img = man.readImageUnsafe(order[cursor]);

            if (fOrder) {
                //EMNIST requires F order to C order
                if (working == null) {
                    working = new byte[28 * 28];
                }
                for (int j = 0; j < 28 * 28; j++) {
                    working[j] = img[28 * (j % 28) + j / 28];
                }
                img = working;
            }

            int label = man.readLabel(order[cursor]);
            if (oneIndexed) {
                //For some inexplicable reason, Emnist LETTERS set is indexed 1 to 26 (i.e., 1 to nClasses), while everything else
                // is indexed (0 to nClasses-1) :/
                label--;
            }

            float[] featureVec = new float[img.length];
            featureData[actualExamples] = featureVec;
            labelData[actualExamples] = new float[numOutcomes];
            labelData[actualExamples][label] = 1.0f;

            for (int j = 0; j < img.length; j++) {
                float v = ((int) img[j]) & 0xFF; //byte is loaded as signed -> convert to unsigned
                if (binarize) {
                    if (v > 30.0f)
                        featureVec[j] = 1.0f;
                    else
                        featureVec[j] = 0.0f;
                } else {
                    featureVec[j] = v / 255.0f;
                }
            }

            actualExamples++;
        }

        if (actualExamples < numExamples) {
            featureData = Arrays.copyOfRange(featureData, 0, actualExamples);
            labelData = Arrays.copyOfRange(labelData, 0, actualExamples);
        }

        INDArray features = Nd4j.create(featureData);
        INDArray labels = Nd4j.create(labelData);
        curr = new DataSet(features, labels);
    }

    @Override
    public void reset() {
        cursor = 0;
        curr = null;
        if (shuffle) {
            if((train && numExamples < NUM_EXAMPLES) || (!train && numExamples < NUM_EXAMPLES_TEST)){
                //Shuffle only first N elements
                if(firstShuffle){
                    MathUtils.shuffleArray(order, rng);
                    firstShuffle = false;
                } else {
                    MathUtils.shuffleArraySubset(order, numExamples, rng);
                }
            } else {
                MathUtils.shuffleArray(order, rng);
            }
        }
    }

    @Override
    public DataSet next() {
        DataSet next = super.next();
        return next;
    }
}
