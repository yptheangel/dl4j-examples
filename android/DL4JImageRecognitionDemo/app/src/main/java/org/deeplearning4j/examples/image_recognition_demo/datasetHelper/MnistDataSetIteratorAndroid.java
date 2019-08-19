package org.deeplearning4j.examples.image_recognition_demo.datasetHelper;

import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

import java.io.IOException;

/**
 * MNIST data set iterator - 60000 training digits, 10000 test digits, 10 classes.
 * Digits have 28x28 pixels and 1 channel (grayscale).<br>
 * Produces data in c-order "flattened" format, with shape {@code [minibatch, 784]}<br>
 * For futher details, see <a href="http://yann.lecun.com/exdb/mnist/">http://yann.lecun.com/exdb/mnist/</a>
 *
 * @author Adam Gibson
 * @modified for android Wilson Choo

 */

/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

public class MnistDataSetIteratorAndroid extends BaseDatasetIterator {
    public MnistDataSetIteratorAndroid(int batch, int numExamples) throws IOException {
        this(batch, numExamples, false);
    }
    public MnistDataSetIteratorAndroid(int batch, int numExamples, boolean binarize) throws IOException {
        this(batch, numExamples, binarize, true, false, 0);
    }
    public MnistDataSetIteratorAndroid(int batchSize, boolean train, int seed) throws IOException {
        this(batchSize, (train ? MnistDataFetcherAndroid.NUM_EXAMPLES : MnistDataFetcherAndroid.NUM_EXAMPLES_TEST), false, train,
                true, seed);
    }
    public MnistDataSetIteratorAndroid(int batch, int numExamples, boolean binarize, boolean train, boolean shuffle,
                                long rngSeed) throws IOException {
        super(batch, numExamples, new MnistDataFetcherAndroid(binarize, train, shuffle, rngSeed, numExamples));
    }
}
