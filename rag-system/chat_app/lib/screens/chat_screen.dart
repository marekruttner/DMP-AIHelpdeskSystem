import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:chat_app/api/api_service.dart';
import 'package:file_picker/file_picker.dart';
import 'package:chat_app/widgets/common_app_bar.dart';

class ChatScreen extends StatefulWidget {
  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController messageController = TextEditingController();
  final List<Map<String, dynamic>> messages = [];
  final ApiService apiService = ApiService();
  final ScrollController _scrollController = ScrollController();

  List<Map<String, dynamic>> chats = [];
  String? currentChatId;
  String? currentChatName;

  // For simulating the AI typing bubble
  Timer? _typingTimer;
  int _typingStep = 0; // cycles the number of dots in "AI is typing..."

  @override
  void initState() {
    super.initState();
    fetchAllChats();
  }

  @override
  void dispose() {
    // Make sure to cancel timer if user leaves screen
    _typingTimer?.cancel();
    super.dispose();
  }

  Future<void> fetchAllChats() async {
    try {
      final chatList = await apiService.getChats();
      setState(() {
        chats = chatList.map<Map<String, dynamic>>((chat) => {
          "chatId": chat['chat_id'],
          "name": "Chat ${chat['chat_id'].substring(0, 6)}",
          "latestMessage": chat['latest_message'] ?? "No messages yet"
        }).toList();
      });
    } catch (e) {
      setState(() {
        messages.add({"message": "Failed to load chats: $e", "isUser": false});
      });
    }
  }

  void loadChatHistory(String chatId, String chatName) async {
    try {
      currentChatId = chatId;
      currentChatName = chatName;

      final history = await apiService.getChatHistory(chatId);
      setState(() {
        messages.clear();

        String currentSpeaker = "";
        StringBuffer currentMessage = StringBuffer();

        for (var convo in history['history']) {
          if (convo.startsWith("User:")) {
            if (currentMessage.isNotEmpty) {
              messages.add({
                "message": currentMessage.toString().trim(),
                "isUser": currentSpeaker == "User",
              });
              currentMessage.clear();
            }
            currentSpeaker = "User";
            currentMessage.write(convo.replaceFirst("User:", "").trim());
          } else if (convo.startsWith("AI:")) {
            if (currentMessage.isNotEmpty) {
              messages.add({
                "message": currentMessage.toString().trim(),
                "isUser": currentSpeaker == "User",
              });
              currentMessage.clear();
            }
            currentSpeaker = "AI";
            currentMessage.write(convo.replaceFirst("AI:", "").trim());
          } else {
            // same speaker, just append
            currentMessage.write("\n$convo");
          }
        }

        // If anything remains
        if (currentMessage.isNotEmpty) {
          messages.add({
            "message": currentMessage.toString().trim(),
            "isUser": currentSpeaker == "User",
          });
        }
      });

      _scrollToBottom();
    } catch (e) {
      setState(() {
        messages.add({"message": "Failed to load chat history: $e", "isUser": false});
      });
    }
  }

  Future<void> sendMessage() async {
    final userMessage = messageController.text.trim();
    if (userMessage.isEmpty) return;

    // 1) Add the user's message to the list
    setState(() {
      messages.add({"message": userMessage, "isUser": true});
    });
    messageController.clear();
    _scrollToBottom();

    // 2) Start the "typing" bubble to simulate AI is typing
    _showTypingBubble();

    bool newChat = currentChatId == null;
    try {
      final response = await apiService.chat(
        userMessage,
        newChat: newChat,
        chatId: currentChatId,
      );
      final botMessage = response['response'] ?? "No response received";

      if (newChat) {
        currentChatId = response['chat_id'];
        currentChatName = "New Chat";
        await fetchAllChats();
      }

      // 3) Remove typing bubble and show the final AI message
      _hideTypingBubble();
      setState(() {
        messages.add({
          "message": botMessage,
          "isUser": false,
          "chatId": currentChatId,
        });
      });
      _scrollToBottom();
    } catch (e) {
      _hideTypingBubble();
      setState(() {
        messages.add({"message": "Error: $e", "isUser": false});
      });
    }
  }

  // Creates a "typing" bubble at the end of the list and starts a timer
  void _showTypingBubble() {
    // Add a bubble that says "AI is typing" (with a text that we will update)
    setState(() {
      messages.add({
        "message": "AI is typing", // initial text
        "isUser": false,
        "isTyping": true,
      });
    });
    _scrollToBottom();

    // Cancel any existing timer
    _typingTimer?.cancel();
    _typingStep = 0;

    // Start a periodic timer that updates the last message's text
    _typingTimer = Timer.periodic(Duration(milliseconds: 500), (timer) {
      // If for some reason the typing bubble has disappeared, stop
      if (messages.isEmpty || messages.last["isTyping"] != true) {
        timer.cancel();
        return;
      }

      // Cycle from 0..3 dots
      _typingStep = (_typingStep + 1) % 4;
      final dots = "." * _typingStep; // "", ".", "..", "..."

      setState(() {
        messages[messages.length - 1]["message"] = "AI is typing$dots";
      });
    });
  }

  // Removes the typing bubble (if present) and stops the timer
  void _hideTypingBubble() {
    // Stop the timer
    _typingTimer?.cancel();
    _typingTimer = null;
    _typingStep = 0;

    // Remove the typing bubble if it is still there
    if (messages.isNotEmpty && messages.last["isTyping"] == true) {
      setState(() {
        messages.removeLast();
      });
    }
  }

  Future<void> embedDocument() async {
    if (currentChatId == null) {
      setState(() {
        messages.add({
          "message": "Cannot embed a document in a new chat. Please send a message first.",
          "isUser": false
        });
      });
      return;
    }

    try {
      FilePickerResult? result = await FilePicker.platform.pickFiles();
      if (result == null) return; // user canceled

      final filePath = result.files.single.path;
      if (filePath == null) return;

      final response = await apiService.uploadDocument(filePath, "chat", chatId: currentChatId!);
      setState(() {
        messages.add({"message": "Document embedded successfully.", "isUser": false});
      });
      _scrollToBottom();
    } catch (e) {
      setState(() {
        messages.add({"message": "Failed to embed document: $e", "isUser": false});
      });
    }
  }

  void startNewChat() {
    setState(() {
      messages.clear();
      currentChatId = null;
      currentChatName = null;
    });
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: Duration(milliseconds: 400),
          curve: Curves.easeOut,
        );
      }
    });
  }

  /// Show a dialog to select 1-5 rating for an AI message.
  void _showRatingDialog(String chatId) {
    if (chatId.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("No valid chatId to rate.")),
      );
      return;
    }

    int selectedRating = 3; // default

    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: Text("Rate AI response"),
          content: StatefulBuilder(
            builder: (BuildContext context, setStateDialog) {
              return Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text("Pick a rating between 1 (worst) and 5 (best):"),
                  SizedBox(height: 10),
                  for (int i = 1; i <= 5; i++)
                    RadioListTile<int>(
                      title: Text('$i'),
                      value: i,
                      groupValue: selectedRating,
                      onChanged: (value) {
                        setStateDialog(() {
                          selectedRating = value!;
                        });
                      },
                    ),
                ],
              );
            },
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text("Cancel"),
            ),
            TextButton(
              onPressed: () async {
                Navigator.pop(context);
                await _submitRating(chatId, selectedRating);
              },
              child: Text("Submit"),
            ),
          ],
        );
      },
    );
  }

  Future<void> _submitRating(String chatId, int rating) async {
    try {
      await apiService.rateResponse(chatId, rating);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Thanks for rating!")),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Rating failed: $e")),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: CommonAppBar(
        apiService: apiService,
        title: currentChatName ?? "Select a Chat",
      ),
      drawer: Drawer(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            DrawerHeader(
              decoration: BoxDecoration(color: Colors.pink),
              child: Text(
                "Chats",
                style: TextStyle(color: Colors.white, fontSize: 24),
              ),
            ),
            ...chats.map((chat) {
              return ListTile(
                title: Text(chat['name']),
                onTap: () {
                  Navigator.pop(context);
                  loadChatHistory(chat['chatId'], chat['name']);
                },
              );
            }).toList(),
            ListTile(
              leading: Icon(Icons.add),
              title: Text("Start New Chat"),
              onTap: () {
                Navigator.pop(context);
                startNewChat();
              },
            ),
          ],
        ),
      ),
      body: Column(
        children: [
          // Main chat area
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              itemCount: messages.length,
              itemBuilder: (context, index) {
                final chatMessage = messages[index];
                return ChatBubble(
                  message: chatMessage['message'],
                  isUser: chatMessage['isUser'],
                  chatId: chatMessage.containsKey('chatId')
                      ? chatMessage['chatId']
                      : null,
                  onRatePressed: (chatMessage['isUser'] == false &&
                      chatMessage['chatId'] != null)
                      ? () => _showRatingDialog(chatMessage['chatId'])
                      : null,
                );
              },
            ),
          ),

          // Message input row
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              children: [
                IconButton(
                  icon: Icon(Icons.add, color: Colors.pink),
                  onPressed: embedDocument,
                ),
                Expanded(
                  child: TextField(
                    controller: messageController,
                    decoration: InputDecoration(
                      hintText: "Type your message...",
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                    ),
                  ),
                ),
                SizedBox(width: 10),
                FloatingActionButton(
                  onPressed: sendMessage,
                  backgroundColor: Colors.pink,
                  child: Icon(Icons.send),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class ChatBubble extends StatelessWidget {
  final String message;
  final bool isUser;
  final String? chatId;
  final VoidCallback? onRatePressed;

  const ChatBubble({
    Key? key,
    required this.message,
    required this.isUser,
    this.chatId,
    this.onRatePressed,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // If it’s a special "typing" bubble, we might see text like "AI is typing..."
    // That is handled by the message text itself in setState updates.

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: EdgeInsets.symmetric(vertical: 5, horizontal: 10),
        padding: EdgeInsets.all(15),
        decoration: BoxDecoration(
          color: isUser ? Colors.pink[100] : Colors.grey[200],
          borderRadius: BorderRadius.circular(15),
        ),
        child: Column(
          crossAxisAlignment:
          isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
          children: [
            // If user message, show normal text; if AI, show Markdown
            isUser
                ? Text(
              message,
              style: TextStyle(fontSize: 16, height: 1.5),
            )
                : MarkdownBody(
              data: message,
              styleSheet: MarkdownStyleSheet(
                p: TextStyle(fontSize: 16, height: 1.5),
                listBullet: TextStyle(fontSize: 16, height: 1.5),
              ),
            ),

            // "Rate" button if it's an AI message (not user) and onRatePressed is provided
            if (!isUser && onRatePressed != null)
              Align(
                alignment: Alignment.bottomRight,
                child: TextButton(
                  onPressed: onRatePressed,
                  child: Text("Rate", style: TextStyle(color: Colors.blue)),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
  