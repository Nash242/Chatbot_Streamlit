# css = '''
# <style>
# .chat-message {
#     padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
# }
# .chat-message.user {
#     background-color: black
# }
# .chat-message.bot {
#     background-color: black
# }
# .chat-message .avatar {
#   width: 20%;
# }
# .chat-message .avatar img {
#   max-width: 78px;
#   max-height: 78px;
#   border-radius: 50%;
#   object-fit: cover;
# }
# .chat-message .message {
#   width: 80%;
#   padding: 0 1.5rem;
#   color: #fff;
# }

# </style>
# '''

# bot_template = '''
# <div class="chat-message bot">
#     <div class="avatar">
#         <img src="https://raw.githubusercontent.com/Nash242/OpenAI_Demo/main/client.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
#     </div>
#     <div class="message">{{MSG}}</div>
# </div>
# '''

# user_template = '''
# <div class="chat-message user">
#     <div class="avatar">
#         <img src="https://raw.githubusercontent.com/Nash242/OpenAI_Demo/main/chatbot.png">
#     </div>    
#     <div class="message">{{MSG}}</div>
# </div>
# '''

# pg_bg_img = """
# <style>
# data-testid="stAppViewContainer"{
# background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT4wb8G9joueCkHnSXdiCRD4x5s8fW4WBucgo0ewQSqQU8R_eFByrikF-o54g&s");
# background-size: cover;
# }
# </style>
# """