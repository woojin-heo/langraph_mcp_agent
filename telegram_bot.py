"""
Telegram Bot for LangGraph + MCP Agent
User ID restricted for security
"""
import os
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters

from langchain_core.messages import HumanMessage, ToolMessage

# Reuse components from agent.py (no duplication!)
from agent import MultiMCPClient, create_graph, SERVERS, TOOLS_REQUIRING_APPROVAL

load_dotenv()

# ============== Configuration ==============

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_IDS = [int(id.strip()) for id in os.getenv("ALLOWED_USER_IDS", "").split(",") if id.strip()]

# ============== Global State ==============

mcp = None
graph = None
user_messages = {}  # user_id -> messages list
pending_approvals = {}  # user_id -> {tool_name, tool_args, tool_call_id, full_result}


# ============== Security ==============

def is_authorized(user_id: int) -> bool:
    """Check if user is authorized"""
    if not ALLOWED_USER_IDS:
        print("⚠️ Warning: ALLOWED_USER_IDS not set. Denying all access.")
        return False
    return user_id in ALLOWED_USER_IDS


async def security_check(update: Update) -> bool:
    """Check authorization and log access attempts"""
    user_id = update.effective_user.id
    user_name = update.effective_user.username or "unknown"
    
    if not is_authorized(user_id):
        print(f"⛔ Unauthorized access attempt: {user_id} (@{user_name})")
        await update.message.reply_text("⛔ Access denied.")
        return False
    
    return True


# ============== Agent Setup ==============

async def init_agent():
    """Initialize MCP client and LangGraph (reusing agent.py)"""
    global mcp, graph
    
    print("🔄 Connecting to MCP servers...")
    mcp = MultiMCPClient()
    await mcp.connect_all(SERVERS)
    
    if not mcp.tools:
        print("❌ No tools available!")
        return False
    
    print(f"✅ Connected! Tools: {[t.name for t in mcp.tools]}")
    
    # Create graph WITHOUT CLI approval (handle approval via Telegram buttons)
    graph = create_graph(mcp, use_cli_approval=False)
    return True


# ============== Telegram Handlers ==============

async def start_command(update: Update, context):
    """Handle /start command"""
    if not await security_check(update):
        return
    
    keyboard = [
        ["📅 Today's schedule", "📅 This week's schedule"],
        ["➕ Add event", "🗺️ Find directions"],
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        "👋 Hello! I am AI assistant for your daily tasks.\n\n"
        "Send a message or click the buttons below!\n\n"
        "Example: 'Tell me about my tomorrow\'s schedule', 'Find the way from Seoul Station to Gangnam Station'\n\n"
        "🔒 Operations that require approval: create/update/delete events.",
        reply_markup=reply_markup
    )


async def help_command(update: Update, context):
    """Handle /help command"""
    if not await security_check(update):
        return
    
    await update.message.reply_text(
        "📖 Usage Guide\n\n"
        "📅 Calendar\n"
        "• Tell me about my today's/tomorrow's/this week's schedule\n"
        "• Add a 'meeting' event for tomorrow at 3pm\n\n"
        "🗺️ Maps\n"
        "• Find restaurants near Gangnam Station\n"
        "• Find the way from Seoul Station to Gangnam Station using public transportation\n\n"
        "⚙️ Commands\n"
        "/start - Start\n"
        "/help - Help\n"
        "/clear - Clear conversation",
    )


async def clear_command(update: Update, context):
    """Handle /clear command - reset conversation"""
    if not await security_check(update):
        return
    
    user_id = update.effective_user.id
    user_messages[user_id] = []
    
    await update.message.reply_text("🗑️ Conversation history has been cleared.")


async def handle_message(update: Update, context):
    """Handle text messages"""
    if not await security_check(update):
        return
    
    user_id = update.effective_user.id
    user_text = update.message.text
    
    # Initialize user messages if needed
    if user_id not in user_messages:
        user_messages[user_id] = []
    
    # Quick menu buttons
    quick_actions = {
        "📅 Today's schedule": "Tell me about my today's schedule",
        "📅 This week's schedule": "Tell me about my this week's schedule",
        "➕ Add event": "I want to add an event. What information do you need?",
        "🗺️ Find directions": "I want to find directions. Please tell me the origin and destination.",
    }
    
    if user_text in quick_actions:
        user_text = quick_actions[user_text]
    
    # Add user message
    user_messages[user_id].append(HumanMessage(content=user_text))
    
    # Show typing indicator
    await update.message.reply_chat_action("typing")
    
    try:
        # Get first LLM response
        result = await graph.ainvoke({"messages": user_messages[user_id]})
        last_message = result["messages"][-1]
        
        # Check if tool requires approval
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tc in last_message.tool_calls:
                if tc["name"] in TOOLS_REQUIRING_APPROVAL:
                    # Store pending approval
                    pending_approvals[user_id] = {
                        "tool_name": tc["name"],
                        "tool_args": tc["args"],
                        "tool_call_id": tc["id"],
                        "full_result": result,
                    }
                    
                    # Send approval request
                    await send_approval_request(update, tc["name"], tc["args"])
                    return
        
        # No approval needed, update messages and send response
        user_messages[user_id] = result["messages"]
        response = result["messages"][-1].content
        
        await update.message.reply_text(response)
        
    except Exception as e:
        print(f"Error: {e}")
        await update.message.reply_text(f"❌ An error occurred: {str(e)[:100]}")


async def send_approval_request(update: Update, tool_name: str, args: dict):
    """Send approval request with inline buttons"""
    keyboard = [
        [
            InlineKeyboardButton("✅ Approve", callback_data="approve"),
            InlineKeyboardButton("❌ Cancel", callback_data="cancel"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Format arguments nicely
    args_text = "\n".join([f"  • {k}: {v}" for k, v in args.items()])
    
    tool_display_names = {
        "create_event": "📅 Create event",
        "update_event": "✏️ Update event",
        "delete_event": "🗑️ Delete event",
    }
    display_name = tool_display_names.get(tool_name, tool_name)
    
    message = (
        f"🔐 **승인 필요: {display_name}**\n\n"
        f"다음 작업을 실행할까요?\n\n"
        f"{args_text}\n\n"
        f"아래 버튼을 눌러 승인하거나 취소하세요."
    )
    
    await update.message.reply_text(
        message,
        reply_markup=reply_markup
    )


async def handle_approval_callback(update: Update, context):
    """Handle approval button clicks"""
    query = update.callback_query
    user_id = query.from_user.id
    
    if not is_authorized(user_id):
        await query.answer("⛔ Access denied.", show_alert=True)
        return
    
    await query.answer()  # Acknowledge button click
    
    if user_id not in pending_approvals:
        await query.edit_message_text("⏰ Approval request has expired.")
        return
    
    pending = pending_approvals.pop(user_id)
    
    if query.data == "approve":
        await query.edit_message_text("⏳ Executing...")
        
        try:
            # Execute the tool
            result = await mcp.call_tool(pending["tool_name"], pending["tool_args"])
            
            # Add tool result to messages
            user_messages[user_id] = pending["full_result"]["messages"]
            user_messages[user_id].append(
                ToolMessage(content=result, tool_call_id=pending["tool_call_id"])
            )
            
            # Get final response from LLM
            final_result = await graph.ainvoke({"messages": user_messages[user_id]})
            user_messages[user_id] = final_result["messages"]
            
            response = final_result["messages"][-1].content
            await query.edit_message_text(f"✅ Completed!\n\n{response}")
            
        except Exception as e:
            await query.edit_message_text(f"❌ Error: {str(e)[:100]}")
    
    else:  # cancel
        await query.edit_message_text("❌ Operation cancelled.")
        
        # Add cancellation to messages
        user_messages[user_id] = pending["full_result"]["messages"]
        user_messages[user_id].append(
            ToolMessage(content="User cancelled the operation.", tool_call_id=pending["tool_call_id"])
        )


# ============== Main ==============

async def post_init(application):
    """Initialize agent after bot starts"""
    success = await init_agent()
    if not success:
        print("❌ Failed to initialize agent!")
        raise SystemExit(1)


def main():
    """Start the Telegram bot"""
    if not TELEGRAM_TOKEN:
        print("TELEGRAM_BOT_TOKEN not set in .env file!")
        print("   Please add: TELEGRAM_BOT_TOKEN=your_token_here")
        return
    
    if not ALLOWED_USER_IDS:
        print("ALLOWED_USER_IDS not set in .env file!")
        print("   Please add: ALLOWED_USER_IDS=your_telegram_id")
        print("   (Get your ID by messaging @userinfobot on Telegram)")
        return
    
    print(f"🔒 Allowed users: {ALLOWED_USER_IDS}")
    print("🤖 Starting Telegram bot...")
    
    # Build application
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(handle_approval_callback))
    
    # Start polling
    print("✅ Bot is running! Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
