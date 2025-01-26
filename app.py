from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the LLM model and tokenizer
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Update this to your chosen LLM model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Responds to the /start command with a welcome message.
    """
    await update.message.reply_text("Hello! I'm your AI Assistant. Ask me anything!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles user messages, processes them with the LLM, and sends a response.
    """
    user_message = update.message.text
    try:
        # Log the user's message
        print(f"Received message: {user_message}")

        # Encode the user's message and generate a response using the LLM
        inputs = tokenizer.encode(user_message, return_tensors="pt")
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Log the model's response
        print(f"Generated response: {response}")

        # Send the LLM's response back to the user
        await update.message.reply_text(response)
    except Exception as e:
        # Handle errors gracefully and inform the user
        await update.message.reply_text("Sorry, I encountered an error while processing your message.")
        print(f"Error: {e}")

if __name__ == "__main__":
    TOKEN = "YOUR_TOKEN"  # Replace with your actual bot token

    # Create the bot application
    app = ApplicationBuilder().token(TOKEN).build()

    # Add command and message handlers
    app.add_handler(CommandHandler("start", start))  # Handles /start command
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))  # Handles text messages

    # Run the bot
    print("Bot is running...")
    app.run_polling()

