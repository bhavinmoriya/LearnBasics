# Complete Reflex Expert Learning Path
## From Zero to Hero

## What is Reflex?

**Reflex** is a pure-Python web framework that lets you build full-stack web apps without writing a single line of JavaScript. It compiles Python to React on the frontend and uses FastAPI on the backend. Think of it as "React but in Python."

**Why Reflex is Booming:**
- Write frontend and backend in pure Python
- No JavaScript/TypeScript required
- Built-in state management
- Component-based architecture like React
- Fast development speed
- Production-ready with deployment support

---

## Phase 1: Foundations (Days 1-3)

### Day 1: Setup & First App

**Install Reflex:**
```bash
pip install reflex
```

**Create your first app:**
```bash
reflex init
```

**Basic "Hello World" (app.py):**
```python
import reflex as rx

def index():
    return rx.text("Hello, World!")

app = rx.App()
app.add_page(index)
```

**Run it:**
```bash
reflex run
```

**🎯 Exercise:** Create a page that displays your name, a greeting, and the current date.

---

### Day 2: Components & Layout

**Understanding Components:**

Components are the building blocks. Reflex has 100+ built-in components.

```python
import reflex as rx

def index():
    return rx.vstack(
        rx.heading("Welcome to My App", size="9"),
        rx.text("This is a paragraph"),
        rx.button("Click Me"),
        rx.image(src="https://picsum.photos/200"),
        spacing="4",
        align="center"
    )

app = rx.App()
app.add_page(index)
```

**Key Layout Components:**
- `rx.vstack()` - vertical stack
- `rx.hstack()` - horizontal stack
- `rx.center()` - center content
- `rx.container()` - responsive container
- `rx.box()` - flexible container

**🎯 Exercise:** Build a landing page with a header, hero section, 3 feature cards in a row, and a footer.

---

### Day 3: State Management

**State is the heart of Reflex.** It's how you make apps interactive.

```python
import reflex as rx

class State(rx.State):
    count: int = 0
    
    def increment(self):
        self.count += 1
    
    def decrement(self):
        self.count -= 1

def index():
    return rx.vstack(
        rx.heading(f"Count: {State.count}"),
        rx.hstack(
            rx.button("Increment", on_click=State.increment),
            rx.button("Decrement", on_click=State.decrement),
        ),
        spacing="4"
    )

app = rx.App()
app.add_page(index)
```

**Key Concepts:**
- State variables are class attributes
- State methods modify the state
- `on_click`, `on_change` connect UI to state
- State updates automatically re-render UI

**🎯 Exercise:** Build a todo app where you can add, remove, and mark todos as complete.

---

## Phase 2: Intermediate Skills (Days 4-7)

### Day 4: Forms & User Input

```python
import reflex as rx

class FormState(rx.State):
    form_data: dict = {}
    
    def handle_submit(self, form_data: dict):
        self.form_data = form_data

def index():
    return rx.vstack(
        rx.form(
            rx.vstack(
                rx.input(name="username", placeholder="Username"),
                rx.input(name="email", type="email", placeholder="Email"),
                rx.text_area(name="message", placeholder="Message"),
                rx.button("Submit", type="submit"),
            ),
            on_submit=FormState.handle_submit,
        ),
        rx.divider(),
        rx.heading("Submitted Data:"),
        rx.text(f"{FormState.form_data}"),
    )

app = rx.App()
app.add_page(index)
```

**Form Components:**
- `rx.input()` - text input
- `rx.text_area()` - multi-line text
- `rx.select()` - dropdown
- `rx.checkbox()` - checkbox
- `rx.radio()` - radio buttons
- `rx.slider()` - slider input

**🎯 Exercise:** Build a contact form with validation (check if email is valid, required fields).

---

### Day 5: Routing & Multi-Page Apps

```python
import reflex as rx

def index():
    return rx.vstack(
        rx.heading("Home Page"),
        rx.link("Go to About", href="/about"),
    )

def about():
    return rx.vstack(
        rx.heading("About Page"),
        rx.link("Go Home", href="/"),
    )

def blog_post():
    return rx.vstack(
        rx.heading(f"Blog Post: {rx.State.router.page.params.get('id', 'N/A')}"),
    )

app = rx.App()
app.add_page(index, route="/")
app.add_page(about, route="/about")
app.add_page(blog_post, route="/blog/[id]")
```

**Routing Features:**
- Static routes: `/about`
- Dynamic routes: `/blog/[id]`
- Query parameters: `rx.State.router.page.params`
- Programmatic navigation: `rx.redirect()`

**🎯 Exercise:** Build a blog with homepage (list of posts), individual post pages (dynamic routes), and an about page.

---

### Day 6: Working with Data & APIs

```python
import reflex as rx
import httpx

class DataState(rx.State):
    posts: list = []
    loading: bool = False
    
    async def fetch_posts(self):
        self.loading = True
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://jsonplaceholder.typicode.com/posts"
            )
            self.posts = response.json()[:5]
        self.loading = False

def index():
    return rx.vstack(
        rx.button("Fetch Posts", on_click=DataState.fetch_posts),
        rx.cond(
            DataState.loading,
            rx.spinner(),
            rx.foreach(
                DataState.posts,
                lambda post: rx.card(
                    rx.heading(post["title"], size="5"),
                    rx.text(post["body"]),
                )
            )
        ),
    )

app = rx.App()
app.add_page(index)
```

**Key Patterns:**
- Use `async` methods for API calls
- `rx.foreach()` to render lists
- `rx.cond()` for conditional rendering
- Loading states for better UX

**🎯 Exercise:** Build a weather app that fetches and displays weather data from a public API.

---

### Day 7: Styling & Theming

```python
import reflex as rx

def index():
    return rx.box(
        rx.heading("Styled Component"),
        background="blue.500",
        color="white",
        padding="4",
        border_radius="lg",
        box_shadow="xl",
        _hover={
            "background": "blue.600",
            "transform": "scale(1.05)",
        },
        transition="all 0.3s",
    )

# Global theme
app = rx.App(
    theme=rx.theme(
        accent_color="blue",
        gray_color="slate",
    )
)
app.add_page(index)
```

**Styling Options:**
1. **Inline styles** (props)
2. **Pseudo-classes** (`_hover`, `_focus`)
3. **Responsive** (`{"base": "value", "md": "value"}`)
4. **Theme tokens** (colors, spacing)

**🎯 Exercise:** Build a dark/light mode toggle with theme switching.

---

## Phase 3: Advanced Techniques (Days 8-12)

### Day 8: Advanced State Management

```python
import reflex as rx

class User(rx.Base):
    username: str
    email: str

class AppState(rx.State):
    users: list[User] = []
    
    def add_user(self, username: str, email: str):
        self.users.append(User(username=username, email=email))
    
    @rx.var
    def user_count(self) -> int:
        return len(self.users)
    
    @rx.var
    def has_users(self) -> bool:
        return len(self.users) > 0

# Computed vars (cached)
# Substates for organization
class AuthState(AppState):
    logged_in: bool = False
    username: str = ""
    
    def login(self, username: str):
        self.logged_in = True
        self.username = username
```

**Advanced Features:**
- **Computed vars** with `@rx.var` (auto-cached)
- **Substates** for modular state
- **Custom types** with `rx.Base`
- **Event chains** (multiple handlers)

**🎯 Exercise:** Build a shopping cart with computed totals, tax calculation, and quantity management.

---

### Day 9: Database Integration

```python
import reflex as rx
from sqlmodel import Field

class User(rx.Model, table=True):
    username: str
    email: str
    created_at: str = Field(default_factory=lambda: str(rx.datetime.now()))

class DBState(rx.State):
    users: list[User] = []
    
    def load_users(self):
        with rx.session() as session:
            self.users = session.exec(
                User.select()
            ).all()
    
    def add_user(self, form_data: dict):
        with rx.session() as session:
            user = User(**form_data)
            session.add(user)
            session.commit()
        self.load_users()

def index():
    return rx.vstack(
        rx.button("Load Users", on_click=DBState.load_users),
        rx.foreach(
            DBState.users,
            lambda user: rx.text(f"{user.username} - {user.email}")
        ),
    )

app = rx.App()
app.add_page(index)
```

**Database Features:**
- Built-in SQLModel integration
- Automatic migrations
- Session management
- Works with SQLite, PostgreSQL, MySQL

**🎯 Exercise:** Build a full CRUD app (Create, Read, Update, Delete) for managing a book library.

---

### Day 10: File Uploads & Media

```python
import reflex as rx

class UploadState(rx.State):
    img: list[str] = []
    
    async def handle_upload(self, files: list[rx.UploadFile]):
        for file in files:
            upload_data = await file.read()
            outfile = f"./uploaded_files/{file.filename}"
            
            with open(outfile, "wb") as f:
                f.write(upload_data)
            
            self.img.append(file.filename)

def index():
    return rx.vstack(
        rx.upload(
            rx.button("Select Files"),
            id="upload1",
            multiple=True,
            accept={"image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"]},
        ),
        rx.button(
            "Upload",
            on_click=UploadState.handle_upload(rx.upload_files(upload_id="upload1")),
        ),
        rx.foreach(
            UploadState.img,
            lambda img: rx.image(src=f"/uploaded_files/{img}", width="200px"),
        ),
    )

app = rx.App()
app.add_page(index)
```

**🎯 Exercise:** Build an image gallery app where users can upload, view, and delete images.

---

### Day 11: Authentication & Authorization

```python
import reflex as rx
import bcrypt

class AuthState(rx.State):
    is_authenticated: bool = False
    username: str = ""
    
    def login(self, username: str, password: str):
        # In real app, check against database
        if username == "admin" and password == "password":
            self.is_authenticated = True
            self.username = username
            return rx.redirect("/dashboard")
        return rx.window_alert("Invalid credentials")
    
    def logout(self):
        self.is_authenticated = False
        self.username = ""
        return rx.redirect("/")
    
    def check_auth(self):
        if not self.is_authenticated:
            return rx.redirect("/login")

def protected_page():
    return rx.vstack(
        rx.heading(f"Welcome {AuthState.username}"),
        rx.button("Logout", on_click=AuthState.logout),
        on_mount=AuthState.check_auth,
    )

def login_page():
    return rx.form(
        rx.vstack(
            rx.input(name="username", placeholder="Username"),
            rx.input(name="password", type="password", placeholder="Password"),
            rx.button("Login", type="submit"),
        ),
        on_submit=lambda form: AuthState.login(form["username"], form["password"]),
    )

app = rx.App()
app.add_page(login_page, route="/login")
app.add_page(protected_page, route="/dashboard")
```

**🎯 Exercise:** Build a complete authentication system with signup, login, protected routes, and user profiles.

---

### Day 12: Real-time Features & WebSockets

```python
import reflex as rx
import asyncio

class ChatState(rx.State):
    messages: list[dict] = []
    current_message: str = ""
    
    def send_message(self):
        if self.current_message:
            self.messages.append({
                "text": self.current_message,
                "timestamp": str(rx.datetime.now()),
            })
            self.current_message = ""
    
    def set_message(self, value: str):
        self.current_message = value

def index():
    return rx.vstack(
        rx.box(
            rx.foreach(
                ChatState.messages,
                lambda msg: rx.text(f"{msg['timestamp']}: {msg['text']}"),
            ),
            height="400px",
            overflow_y="scroll",
            border="1px solid gray",
            padding="4",
        ),
        rx.hstack(
            rx.input(
                value=ChatState.current_message,
                on_change=ChatState.set_message,
                placeholder="Type a message...",
            ),
            rx.button("Send", on_click=ChatState.send_message),
        ),
    )

app = rx.App()
app.add_page(index)
```

**🎯 Exercise:** Build a real-time chat application with message history and timestamps.

---

## Phase 4: Production & Best Practices (Days 13-14)

### Day 13: Performance Optimization

**Best Practices:**

1. **Use computed vars for expensive calculations:**
```python
@rx.var
def expensive_calculation(self) -> int:
    # Only recomputes when dependencies change
    return sum(item.price for item in self.cart_items)
```

2. **Lazy load data:**
```python
def on_mount_handler(self):
    if not self.data_loaded:
        self.fetch_data()
```

3. **Debounce user input:**
```python
rx.input(
    on_change=State.set_search,
    debounce_timeout=500,  # Wait 500ms after typing stops
)
```

4. **Paginate large lists:**
```python
@rx.var
def visible_items(self) -> list:
    start = self.page * self.page_size
    end = start + self.page_size
    return self.all_items[start:end]
```

**🎯 Exercise:** Optimize a slow app with large data by implementing pagination and computed vars.

---

### Day 14: Deployment

**Deploy to Reflex Cloud (Easiest):**
```bash
reflex deploy
```

**Self-hosted (Docker):**
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN reflex init
RUN reflex export --frontend-only
CMD reflex run --env prod
```

**Environment Variables:**
```python
# rxconfig.py
import reflex as rx

config = rx.Config(
    app_name="myapp",
    db_url=os.getenv("DATABASE_URL", "sqlite:///reflex.db"),
)
```

**Production Checklist:**
- [ ] Set up database (PostgreSQL recommended)
- [ ] Configure environment variables
- [ ] Enable HTTPS
- [ ] Set up error monitoring
- [ ] Configure CORS if needed
- [ ] Optimize assets (images, fonts)
- [ ] Set up CI/CD pipeline

**🎯 Exercise:** Deploy your todo app or portfolio to production.

---

## Real-World Project Ideas

Now build these to cement your skills:

1. **Personal Dashboard** - Weather, calendar, todos, notes
2. **E-commerce Store** - Products, cart, checkout, admin panel
3. **Social Media Clone** - Posts, likes, comments, user profiles
4. **Analytics Dashboard** - Charts, graphs, data visualization
5. **Project Management Tool** - Kanban board, tasks, teams
6. **Portfolio Website** - Showcase your projects
7. **Blog Platform** - Write, edit, publish articles
8. **Real-time Chat App** - Messages, rooms, notifications
9. **Expense Tracker** - Track spending, categories, reports
10. **Recipe Manager** - Save recipes, meal planning

---

## Essential Resources

**Official:**
- Documentation: https://reflex.dev/docs
- GitHub: https://github.com/reflex-dev/reflex
- Discord Community: Join for help
- Component Library: https://reflex.dev/docs/library

**Learning:**
- YouTube tutorials
- Build projects from docs examples
- Read source code of example apps
- Join community discussions

---

## Pro Tips

1. **Start small, iterate fast** - Don't build everything at once
2. **Read the docs thoroughly** - They're excellent
3. **Use the component library** - Don't reinvent the wheel
4. **State management is key** - Master it early
5. **Think in components** - Break UI into reusable pieces
6. **Test in production early** - Deploy quickly, iterate
7. **Join the community** - Discord is super helpful
8. **Python skills transfer** - If you know Python, you're 80% there

---

## 14-Day Challenge Summary

- **Days 1-3:** Setup, components, basic state
- **Days 4-7:** Forms, routing, APIs, styling
- **Days 8-12:** Advanced state, databases, auth, real-time
- **Days 13-14:** Performance, deployment

**After 14 days, you'll be able to:**
✅ Build full-stack web apps in pure Python
✅ Handle state management confidently
✅ Work with databases and APIs
✅ Implement authentication and authorization
✅ Deploy production-ready applications
✅ Optimize for performance
✅ Build real-time features

**Next Level (Beyond Expert):**
- Contribute to Reflex itself
- Build custom components
- Create Reflex templates/boilerplates
- Write tutorials for others
- Build SaaS products with Reflex

---

## Quick Reference Cheatsheet

```python
# Components
rx.text(), rx.heading(), rx.button(), rx.input()
rx.vstack(), rx.hstack(), rx.center(), rx.box()

# State
class State(rx.State):
    var: type = default
    @rx.var
    def computed(self) -> type: ...
    def method(self): ...

# Events
on_click, on_change, on_submit, on_mount

# Conditionals & Loops
rx.cond(condition, true_case, false_case)
rx.foreach(list, lambda item: ...)

# Routing
app.add_page(component, route="/path")
rx.link("Text", href="/path")
rx.redirect("/path")

# Styling
component(prop="value", _hover={}, _focus={})
```

Now go build something amazing! 🚀