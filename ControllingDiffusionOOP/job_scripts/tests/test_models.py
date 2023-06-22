
# import stable diffusion model(s)
import models.stable_diffusion as stable_diffusion
import models.clip_extension as clip_extension

# import representations
from representations.prompt import Prompt
from representations.image_noise import ImageNoise
from representations.selection import Selection

# import test packages
import unittest



class TestModelsAndRepresentations(unittest.TestCase):
    def test_stable_diffusion(self) -> None:
        """Test main representations"""
        prompt: Prompt = Prompt("a photograph of an astronaut riding a horse")
        self.assertTrue(prompt.text is not None)
        self.assertTrue(prompt.embeddings is not None)

        rand: ImageNoise = ImageNoise()
        self.assertTrue(rand.seed is None)
        self.assertTrue(rand.image is None)
        self.assertTrue(rand.latents is not None)

        seed: ImageNoise = ImageNoise(seed = 100)
        self.assertTrue(seed.seed is not None)
        self.assertTrue(seed.image is None)
        self.assertTrue(seed.latents is not None)



        """Test the diffusion module"""
        # test the text to image pipeline
        img_rand: ImageNoise = stable_diffusion.diffusion_steps(prompt, rand, num_steps = 1)
        self.assertTrue(img_rand.seed is None)
        self.assertTrue(img_rand.image is not None)
        self.assertTrue(img_rand.latents is not None)
        img_rand.image.save("test_img_rand.png") # save for visual inspection (assert does not really work here, note that num_steps = 1 i.e. wont be good)

        img: ImageNoise = stable_diffusion.diffusion_steps(prompt, seed, num_steps = 1)
        self.assertTrue(img.seed is None)
        self.assertTrue(img.image is not None)
        self.assertTrue(img.latents is not None)
        img.image.save("test_img.png") # save for visual inspection



        """Test the CLIP extension module"""
        CLIP_similarity_score = clip_extension.get_similarity_score(prompt, img_rand)
        print(CLIP_similarity_score) # "save" for manual inspection (not really need but may be useful)



        """Test extra representations"""
        Selection(img_rand.image, [img.image], "label") # normally "label" is the prompt but it has to have the same number of words as the images



if __name__ == '__main__':
    unittest.main()