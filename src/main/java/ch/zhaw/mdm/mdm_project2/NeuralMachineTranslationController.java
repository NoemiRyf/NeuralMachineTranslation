package ch.zhaw.mdm.mdm_project2;

import ai.djl.ModelException;
import ai.djl.translate.TranslateException;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.io.IOException;

@Controller
public class NeuralMachineTranslationController {

    @Autowired
    private NeuralMachineTranslationService translation;

    @GetMapping("/")
    public String index() {
        return "index";
    }

    @PostMapping("/translate")
    public String translate(
            @RequestParam(name = "frenchText", required = false) String frenchText,
            Model model) {
        try {
            String englishText = translation.translateText(frenchText);
            model.addAttribute("englishText", englishText);
        } catch (ModelException | TranslateException | IOException e) {
            model.addAttribute("error", "Translation failed.");
        }
        return "index";
    }
}
