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

    /**
     * Handles the GET request for the root URL ("/").
     * @return The name of the view template to render (index.html in this case).
     */
    @GetMapping("/")
    public String index() {
        return "index";
    }

    /**
     * Handles the POST request for the "/translate" URL.
     * @param frenchText The French text to translate.
     * @param model The model object to add attributes for rendering the view.
     * @return The name of the view template to render (index.html in this case).
     */
    @PostMapping("/translate")
    public String translate(
            @RequestParam(name = "frenchText", required = false) String frenchText,
            Model model) {
        try {
            // Translate the French text to English using the translation service.
            String englishText = translation.translateText(frenchText);
            
            // Add the translated English text as an attribute to the model for rendering.
            model.addAttribute("englishText", englishText);
        } catch (ModelException | TranslateException | IOException e) {
            // Handle any exceptions that occur during translation and add an error attribute to the model.
            model.addAttribute("error", "Translation failed.");
        }
        
        // Return the name of the view template to render (index.html in this case).
        return "index";
    }
}
