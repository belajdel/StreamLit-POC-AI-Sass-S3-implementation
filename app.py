import streamlit as st
import replicate
import random
import time
import os
import re
from PIL import Image
import io
import requests
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import tempfile
import subprocess

# Function for the chat model interface


def geb():
    st.title("Welly ChatBot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "saved_conversations" not in st.session_state:
        st.session_state.saved_conversations = {}

    # Define available models
    models = {
        "Llama 405B": "meta/meta-llama-3.1-405b-instruct"
        # "mamba 2.8B":'adirik/mamba-2.8b-chat:54995daa413e1d85f27126266b8414fbc71fc879368fff2dc7cbfea60b87de31',
    }

    # Function to format messages
    def format_message(content):
        # Escape HTML characters
        content = (
            content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

        # Format code blocks, ensuring they are only formatted when they are actual code
        code_block_pattern = r"```(.*?)```"
        formatted_content = re.sub(
            code_block_pattern, r"<pre><code>\1</code></pre>", content, flags=re.S
        )

        # Format ordered lists
        list_pattern = r"^(\d+)\.\s(.*)$"
        formatted_content = re.sub(
            list_pattern, r"<li>\2</li>", formatted_content, flags=re.MULTILINE
        )

        # Wrap in <ol> if any list items are found
        if "<li>" in formatted_content:
            formatted_content = f"<ol>{formatted_content}</ol>"

        # Return formatted content
        return f"{formatted_content}"

    # JavaScript function to copy text to clipboard
    st.markdown(
        """
    <script>
    function copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(function() {
            alert('Copied to clipboard!');
        }, function(err) {
            console.error('Could not copy text: ', err);
        });
    }
    </script>
    """,
        unsafe_allow_html=True,
    )

    # Model selection
    selected_model = st.selectbox("Choose a model:", list(models.keys()))

    # Sidebar for saving conversations
    st.sidebar.title("Conversation Management")

    # Button to save current conversation
    if st.sidebar.button("Save Conversation"):
        conversation_name = st.sidebar.text_input("Enter a name for this conversation:")
        if conversation_name:
            st.session_state.saved_conversations[
                conversation_name
            ] = st.session_state.messages
            st.sidebar.success(f"Conversation '{conversation_name}' saved!")

    # Display saved conversations
    st.sidebar.subheader("Saved Conversations")
    conversation_names = list(st.session_state.saved_conversations.keys())
    selected_conversation = st.sidebar.selectbox(
        "Select a conversation to load:", [""] + conversation_names
    )

    if selected_conversation:
        st.session_state.messages = st.session_state.saved_conversations[
            selected_conversation
        ]

    # Button to clear messages
    if st.sidebar.button("Clear Messages"):
        st.session_state.messages = []
        st.sidebar.success("Messages cleared!")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                formatted_message = format_message(message["content"])
                st.markdown(formatted_message, unsafe_allow_html=True)
            else:
                st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input(""):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            output = replicate.run(
                models[selected_model],  # Use the selected model
                input={
                    "prompt": f"{prompt}:",
                    "temperature": 0.9,
                    "top_p": 0.9,
                    "max_length": 512,
                    "repetition_penalty": 1,
                },
            )

            # Simulate stream of response with milliseconds delay
            full_response = ""
            for item in output:
                full_response += item + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


def image_generation_interface():
    st.header("Image Generation")

    # Text area for image description
    prompt = st.text_area("Describe the image you want to generate:")

    # Option to modify number of outputs
    num_outputs = st.number_input(
        "Number of images to generate:", min_value=1, max_value=5, value=1
    )

    # Button to generate image
    if st.button("Generate Image"):
        st.write("Generating image based on your description...")

        # Generate image using Replicate
        output = replicate.run(
            "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
            input={
                "width": 768,
                "height": 768,
                "prompt": prompt,
                "scheduler": "K_EULER",
                "num_outputs": num_outputs,
                "guidance_scale": 7.5,
                "num_inference_steps": 50,
            },
        )

        # Check if output is a list of URLs or base64 strings
        if isinstance(output, list):
            for i, img_data in enumerate(output):
                # Assuming img_data is a URL, fetch the image
                image = Image.open(requests.get(img_data, stream=True).raw)

                # Display image in a canvas
                st.image(
                    image, caption=f"Generated Image {i + 1}", use_column_width=True
                )

                # Create a download button
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                buffered.seek(0)
                st.download_button(
                    label=f"Download Image {i + 1}",
                    data=buffered,
                    file_name=f"generated_image_{i + 1}.png",
                    mime="image/png",
                )
        else:
            st.error("Failed to generate images. Please check your input.")


def upload_file_to_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket and return the file URL.

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified, file_name is used
    :return: URL of the uploaded file if successful, else None
    """
    if object_name is None:
        object_name = file_name

    s3_client = boto3.client("s3")

    try:
        s3_client.upload_file(
            file_name, bucket, object_name, ExtraArgs={"ACL": "public-read"}
        )
        print(f"Upload Successful: {file_name} to {bucket}/{object_name}")

        # Construct the URL of the uploaded file
        file_url = f"https://{bucket}.s3.amazonaws.com/{object_name}"
        print(file_url)
        return file_url
    except FileNotFoundError:
        print(f"The file was not found: {file_name}")
        return None
    except NoCredentialsError:
        print("Credentials not available")
        return None
    except ClientError as e:
        print(f"Failed to upload {file_name} to {bucket}/{object_name}: {e}")
        return None


# Example usage


def video_captioning_interface():
    st.header("Video Captioning Interface")

    # Create uploads directory if it doesn't exist
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    # Upload video file
    video_file = st.file_uploader("Upload your video file:", type=["mp4", "mov"])

    if st.button("Caption Video"):
        if video_file is not None:
            st.write("Captioning video...")

            # Save the uploaded video to the uploads directory
            video_path = os.path.join(uploads_dir, video_file.name)
            with open(video_path, "wb") as f:
                f.write(video_file.read())

            # Upload the video to S3
            bucket_name = "sassv2"  # Replace with your bucket name
            object_name = video_file.name  # You can customize the object name
            video_url = upload_file_to_s3(video_path, bucket_name, object_name)

            if video_url:
                # Prepare input for Replicate model
                input_data = {"video_file_input": video_url}  # Use the S3 URL

                # Call the Replicate model for video captioning
                result = replicate.run(
                    "fictions-ai/autocaption:18a45ff0d95feb4449d192bbdc06b4a6df168fa33def76dfc51b78ae224b599b",
                    input=input_data,
                )

                # Assuming 'result' contains the URL of the captioned video
                st.write("Video captioning completed!")
                res = str(result[0])
                st.video(res)  # Display the captioned video
                # Provide a download button for the generated video
                st.download_button(
                    label="Download Captioned Video",
                    data=res,  # Use the URL or data needed for download
                    file_name="captioned_video.mp4",
                    mime="video/mp4",
                )


# Main app
def main():
    st.title("AI Services for Tunisians")
    st.sidebar.header("Navigate to Services")
    # Create a sidebar navigation
    service = st.sidebar.selectbox(
        "Choose a service:",
        ["Home", "Text Generation", "Image Generation", "Video Editing"],
    )

    if service == "Home":
        st.write(
            "Welcome to our SaaS platform offering AI services tailored for the Tunisian market."
        )
        st.write("Choose a service from the sidebar to begin.")
    elif service == "Text Generation":
        geb()
    elif service == "Image Generation":
        image_generation_interface()
    elif service == "Video Editing":
        video_captioning_interface()


if __name__ == "__main__":
    main()
