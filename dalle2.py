import os
import requests
from io import BytesIO
from openai import OpenAI
from PIL import Image

class DalleClient:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.static_description = {
            "ya": "A refined, classical East Asian ink painting on a tan silk background, depicting a dark, brownish-black waterbird—perhaps similar to a duck or cormorant—bending its head gracefully to preen its feathers. The background is minimal and empty, with subtle fiber textures of the silk showing through. The brushwork is delicate and layered, capturing fine details in the bird’s plumage and conveying a quiet, elegant atmosphere reminiscent of ancient ink wash art.",
            "shitao": "A traditional Chinese ink wash painting on a pale, textured paper background, depicting a scholarly figure in a long, flowing robe with loosely tied hair, standing beside a slender tree trunk. The figure leans slightly and appears to hold or smell a chrysanthemum blossom, evoking a sense of quiet contemplation and poetic elegance. Black ink brushstrokes are minimal yet expressive, with a few vertical lines suggesting hanging branches or leaves. Delicate calligraphy and red seals adorn the left side, adding a subtle narrative and artistic authenticity to the scene.",
            "nai": "A minimalistic ink and watercolor-style painting featuring a small, round-bodied bird perched on a slender, black brushstroke branch. The bird’s soft brown and white plumage is suggested with gentle, understated coloration, while its dark beak and expressive eye are defined by simple yet deliberate ink lines. The background is composed of pale, muted washes in off-white and light grays, with subtle hints of abstract shapes—possibly distant cliffs or foliage—evoked by fluid, delicate brushwork. The overall mood is tranquil and refined, emphasizing negative space and the quiet presence of nature.",
            "hs": "A traditional Chinese ink and wash painting on a white background, featuring an elderly figure with a kind, smiling face, dressed in flowing, layered robes of soft grays. He stands beside a large, irregularly shaped rock formation, leaning slightly on a slender staff. To the left, elegant calligraphy and stamped red seals provide a vertical accent, imbuing the scene with poetic significance. Near the bottom, a cluster of small, reddish brushstrokes suggests the presence of fish or drifting leaves, adding a subtle narrative element. The overall style is refined yet understated, employing expressive brushwork and minimal color to convey a sense of tranquility and cultivated wisdom.",
            "fzt": "A traditional Chinese ink and watercolor scene depicting a scholarly figure reclining on a small wooden raft floating in a tranquil river. He rests his head on one hand, gazing thoughtfully toward the far shore. Behind him, a gnarled tree with subtle foliage arches over the water’s edge, while clusters of reeds and grasses grow near the banks. The muted, earthy color palette and delicate brushstrokes evoke a sense of quiet reflection. Elegant calligraphy and red seals appear along the left side, contributing poetic and cultural depth. The composition exudes serenity, scholarship, and a gentle connection with nature.",
            "ddhj": "A traditional Chinese ink and brush painting on a brownish, aged silk fabric, bordered by ornate, floral-patterned textiles. The scene features a lone fisherman seated in a small wooden boat, leaning forward with a simple fishing implement extended over gently rippling water. The composition is minimalist, with soft brown tones and delicate linework suggesting the calm, expansive surface of the water and the quiet concentration of the figure. The overall impression is serene, evoking a timeless moment of solitude and connection with nature."
        }

    def _enhance_prompt_with_llm(self, original_description, user_prompt):
        """使用LLM增强用户的prompt"""
        system_prompt = """You are an expert at writing prompts for DALL-E 2 image inpainting. 
        Your task is to analyze the user's description and create an appropriate prompt for filling in the blank/empty spaces in traditional Chinese ink paintings.
        
        There are two types of user descriptions:
        1. Overall description: User describes the desired final look of the entire artwork
        2. Local description: User specifically describes what should appear in the blank space
        
        For overall descriptions:
        - Extract the relevant elements that should appear in the blank space
        - Ensure these elements match the overall composition
        - Maintain consistency with the existing artwork
        
        For local descriptions:
        - Focus on the specific elements to be added
        - Ensure they blend seamlessly with the surrounding artwork
        - Add style-related context to maintain artistic coherence
        
        Your output prompt should:
        1. Clearly describe what should appear in the blank space
        2. Match the style of traditional ink painting
        3. Ensure visual harmony with the existing elements
        4. Use specific, DALL-E 2 friendly language
        
        Output only the enhanced prompt, nothing else."""

        user_message = f"""Original artwork description: {original_description}
        User's description: {user_prompt}
        
        Create an appropriate prompt for filling the blank space while maintaining artistic coherence."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in prompt enhancement: {e}")
            # 如果LLM调用失败，返回一个基础的组合prompt
            return f"In the style of a traditional ink painting, seamlessly integrate {user_prompt} into the blank space, maintaining harmony with the existing artwork which {original_description}"

    def inpaint(self, image_path, prompt):
        # 提取图像路径中的关键字并获取原始描述
        for key in self.static_description.keys():
            if key in image_path:
                original_description = self.static_description[key]
                # 使用LLM增强prompt
                enhanced_prompt = self._enhance_prompt_with_llm(original_description, prompt)
                break
        else:
            enhanced_prompt = prompt  # 如果没找到匹配的描述，使用原始prompt
        
        print(f"Enhanced prompt: {enhanced_prompt}")
        
        input_image = Image.open(image_path).convert("RGBA")
        original_size = input_image.size
        print(f"Original size: {original_size}")
        
        # Generate the mask path
        # Extract the filename from the path and add 'edited_' prefix
        filename = image_path.split('/')[-1]
        dirname = os.path.dirname(image_path)
        mask_path = os.path.join(dirname, "edited_" + filename)
        print(f"Mask path: {mask_path}")
        mask_image = Image.open(mask_path).convert("RGBA")

        # Resize images to 1024x1024 for DALL-E
        input_image = input_image.resize((1024, 1024))
        mask_image = mask_image.resize((1024, 1024))

        # Save resized images
        input_image.save("resized_image.png")
        mask_image.save("resized_mask.png")

        # Call DALL-E for inpainting
        response = self.client.images.edit(
            model="dall-e-2",
            image=open("resized_image.png", 'rb'),
            mask=open("resized_mask.png", 'rb'),
            prompt=enhanced_prompt,
            n=1,
            size="1024x1024"
        )
        url = response.data[0].url

        # Download the generated image
        response = requests.get(url)  # type: ignore
        if response.status_code == 200:
            # Open the image and resize it back to the original size
            image = Image.open(BytesIO(response.content))
            resized_image = image.resize(original_size)
            
            # Save the generated image
            resized_image.save("generated_image.png")
            print("Generated image saved as 'generated_image.png'")
            return resized_image
        else:
            raise Exception("Error downloading the generated image")