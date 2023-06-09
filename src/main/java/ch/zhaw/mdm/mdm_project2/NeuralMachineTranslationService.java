package ch.zhaw.mdm.mdm_project2;

import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.modality.nlp.preprocess.PunctuationSeparator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.ParameterStore;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;

import com.google.gson.reflect.TypeToken;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Type;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;

@Component
public class NeuralMachineTranslationService {

    private static final Logger logger = LoggerFactory.getLogger(NeuralMachineTranslationService.class);

    private static final int HIDDEN_SIZE = 256;
    private static final int EOS_TOKEN = 1;
    private static final int MAX_LENGTH = 50;

    public String translateText(String frenchText) throws ModelException, TranslateException, IOException {
        Path path = Paths.get("src/main/resources/source_wrd2idx.json");
        Map<String, Long> wrd2idx;
        try (InputStream is = Files.newInputStream(path)) {
            String json = Utils.toString(is);
            Type mapType = new TypeToken<Map<String, Long>>() {}.getType();
            wrd2idx = JsonUtils.GSON.fromJson(json, mapType);
        }

        path = Paths.get("src/main/resources/target_idx2wrd.json");
        Map<String, String> idx2wrd;
        try (InputStream is = Files.newInputStream(path)) {
            String json = Utils.toString(is);
            Type mapType = new TypeToken<Map<String, String>>() {}.getType();
            idx2wrd = JsonUtils.GSON.fromJson(json, mapType);
        }

        Engine engine = Engine.getEngine("PyTorch");
        try (NDManager manager = engine.newBaseManager()) {
            try (ZooModel<NDList, NDList> encoder = getEncoderModel();
                 ZooModel<NDList, NDList> decoder = getDecoderModel()) {

                NDList toDecode = predictEncoder(frenchText, encoder, wrd2idx, manager);
                String englishText = predictDecoder(toDecode, decoder, idx2wrd, manager);

                logger.info("French: {}", frenchText);
                logger.info("English: {}", englishText);

                return englishText;
            }
        }
    }

    public ZooModel<NDList, NDList> getEncoderModel() throws ModelException, IOException {
        String url = "https://resources.djl.ai/demo/pytorch/android/neural_machine_translation/optimized_encoder_150k.zip";

        Criteria<NDList, NDList> criteria = Criteria.builder()
                .setTypes(NDList.class, NDList.class)
                .optModelUrls(url)
                .optModelName("optimized_encoder_150k.ptl")
                .optEngine("PyTorch")
                .build();
        return criteria.loadModel();
    }

    public ZooModel<NDList, NDList> getDecoderModel() throws ModelException, IOException {
        String url = "https://resources.djl.ai/demo/pytorch/android/neural_machine_translation/optimized_decoder_150k.zip";

        Criteria<NDList, NDList> criteria = Criteria.builder()
                .setTypes(NDList.class, NDList.class)
                .optModelUrls(url)
                .optModelName("optimized_decoder_150k.ptl")
                .optEngine("PyTorch")
                .build();
        return criteria.loadModel();
    }

    public NDList predictEncoder(
            String text,
            ZooModel<NDList, NDList> model,
            Map<String, Long> wrd2idx,
            NDManager manager) {
        // maps french input to id's from french file
        List<String> list = Collections.singletonList(text);
        PunctuationSeparator punc = new PunctuationSeparator();
        list = punc.preprocess(list);
        List<Long> inputs = new ArrayList<>();
        for (String word : list) {
            if (word.length() == 1 && !Character.isAlphabetic(word.charAt(0))) {
                continue;
            }
            Long id = wrd2idx.get(word.toLowerCase(Locale.FRENCH));
            if (id == null) {
                throw new IllegalArgumentException("Word \"" + word + "\" not found.");
            }
            inputs.add(id);
        }

        // for forwarding the model
        Shape inputShape = new Shape(1);
        Shape hiddenShape = new Shape(1, 1, 256);
        FloatBuffer fb = FloatBuffer.allocate(256);
        NDArray hiddenTensor = manager.create(fb, hiddenShape);
        long[] outputsShape = {MAX_LENGTH, HIDDEN_SIZE};
        FloatBuffer outputTensorBuffer = FloatBuffer.allocate(MAX_LENGTH * HIDDEN_SIZE);

        // for using the model
        Block block = model.getBlock();
        ParameterStore ps = new ParameterStore();

        // loops through forwarding of each word
        for (long input : inputs) {
            NDArray inputTensor = manager.create(new long[] {input}, inputShape);
            NDList inputTensorList = new NDList(inputTensor, hiddenTensor);
            NDList outputs = block.forward(ps, inputTensorList, false);
            NDArray outputTensor = outputs.get(0);
            outputTensorBuffer.put(outputTensor.toFloatArray());
            hiddenTensor = outputs.get(1);
        }
        outputTensorBuffer.rewind();
        NDArray outputsTensor = manager.create(outputTensorBuffer, new Shape(outputsShape));

        return new NDList(outputsTensor, hiddenTensor);
    }

    public String predictDecoder(
            NDList toDecode,
            ZooModel<NDList, NDList> model,
            Map<String, String> idx2wrd,
            NDManager manager) {
        // for forwarding the model
        Shape decoderInputShape = new Shape(1, 1);
        NDArray inputTensor = manager.create(new long[] {0}, decoderInputShape);
        ArrayList<Integer> result = new ArrayList<>(MAX_LENGTH);
        NDArray outputsTensor = toDecode.get(0);
        NDArray hiddenTensor = toDecode.get(1);

        // for using the model
        Block block = model.getBlock();
        ParameterStore ps = new ParameterStore();

        // loops through forwarding of each word
        for (int i = 0; i < MAX_LENGTH; i++) {
            NDList inputTensorList = new NDList(inputTensor, hiddenTensor, outputsTensor);
            NDList outputs = block.forward(ps, inputTensorList, false);
            NDArray outputTensor = outputs.get(0);
            hiddenTensor = outputs.get(1);
            float[] buf = outputTensor.toFloatArray();
            int topIdx = 0;
            double topVal = -Double.MAX_VALUE;
            for (int j = 0; j < buf.length; j++) {
                if (buf[j] > topVal) {
                    topVal = buf[j];
                    topIdx = j;
                }
            }
            if (topIdx == EOS_TOKEN) {
                break;
            }
            result.add(topIdx);
            inputTensor = manager.create(new long[] {topIdx}, decoderInputShape);
        }

        StringBuilder sb = new StringBuilder();
        for (int i : result) {
            sb.append(idx2wrd.get(Integer.toString(i)));
            sb.append(" ");
        }
        return sb.toString().trim();
    }
}
