from image_captioning import ImageCaptioner
from visual_qa import VisualQA
from image_analysis import ImageAnalyzer
from conversational_vlm import ConversationalVLM
from multi_image_reasoning import MultiImageReasoner

def demo_captioning():
    print("\n" + "="*60)
    print("IMAGE CAPTIONING DEMO")
    print("="*60)
    
    captioner = ImageCaptioner()
    image = "https://images.unsplash.com/photo-1574158622682-e40e69881006"
    
    caption = captioner.generate_caption(image)
    print(f"\nCaption: {caption}")

def demo_vqa():
    print("\n" + "="*60)
    print("VISUAL QA DEMO")
    print("="*60)
    
    vqa = VisualQA()
    image = "https://images.unsplash.com/photo-1574158622682-e40e69881006"
    
    questions = [
        "What is the main subject of this image?",
        "What colors are prominent?",
        "What is the setting or environment?"
    ]
    
    for question in questions:
        answer = vqa.answer_question(image, question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")

def demo_analysis():
    print("\n" + "="*60)
    print("IMAGE ANALYSIS DEMO")
    print("="*60)
    
    analyzer = ImageAnalyzer()
    image = "https://images.unsplash.com/photo-1574158622682-e40e69881006"
    
    print("\n1. Object Detection:")
    print(analyzer.detect_objects(image))
    
    print("\n2. Scene Description:")
    print(analyzer.describe_scene(image))
    
    print("\n3. Color Analysis:")
    print(analyzer.identify_colors(image))

def demo_conversation():
    print("\n" + "="*60)
    print("CONVERSATIONAL VLM DEMO")
    print("="*60)
    
    vlm = ConversationalVLM()
    image = "https://images.unsplash.com/photo-1574158622682-e40e69881006"
    
    turns = [
        "What do you see in this image?",
        "What is the mood or atmosphere?",
        "Can you describe it more poetically?"
    ]
    
    for i, message in enumerate(turns, 1):
        reset = (i == 1)
        response = vlm.chat(image, message, reset_history=reset)
        print(f"\nTurn {i}")
        print(f"User: {message}")
        print(f"Assistant: {response}")

def demo_multi_image():
    print("\n" + "="*60)
    print("MULTI-IMAGE REASONING DEMO")
    print("="*60)
    
    reasoner = MultiImageReasoner()
    image1 = "https://images.unsplash.com/photo-1574158622682-e40e69881006"
    image2 = "https://images.unsplash.com/photo-1543466835-00a7907e9de1"
    
    print("\nComparing two images...")
    comparison = reasoner.compare_images(image1, image2)
    print(comparison)

def main():
    print("\n" + "="*60)
    print("VLM UNIVERSAL IMPLEMENTATION - EXAMPLES")
    print("="*60)
    print("\nSelect a demo:")
    print("1. Image Captioning")
    print("2. Visual Question Answering")
    print("3. Image Analysis")
    print("4. Conversational VLM")
    print("5. Multi-Image Reasoning")
    print("6. Run All Demos")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-6): ")
    
    demos = {
        "1": demo_captioning,
        "2": demo_vqa,
        "3": demo_analysis,
        "4": demo_conversation,
        "5": demo_multi_image,
    }
    
    if choice == "6":
        for demo in demos.values():
            demo()
    elif choice in demos:
        demos[choice]()
    elif choice == "0":
        print("Exiting...")
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()

