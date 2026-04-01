/// A sample struct
#[derive(Debug, Clone)]
pub struct MyStruct {
    pub name: String,
    age: u32,
}

/// Implementation for MyStruct
impl MyStruct {
    /// Creates a new instance
    pub fn new(name: String, age: u32) -> Self {
        Self { name, age }
    }

    /// Gets the name
    pub fn name(&self) -> &str {
        &self.name
    }

    pub async fn fetch(&self) -> Result<Data, Error> {
        todo!()
    }

    pub unsafe fn raw_ptr(&self) -> *const u8 {
        std::ptr::null()
    }
}

/// A trait definition
pub trait Drawable {
    fn draw(&self);
    fn bounds(&self) -> Rect;
}

/// Implement Drawable for MyStruct
impl Drawable for MyStruct {
    fn draw(&self) {
        println!("drawing");
    }

    fn bounds(&self) -> Rect {
        Rect::default()
    }
}

/// A standalone function
pub fn greet(name: &str) -> String {
    format!("Hello, {}", name)
}

/// A generic function
pub fn process<T: Display + Debug>(item: T) -> Result<(), Error> {
    Ok(())
}

/// An enum
#[derive(Debug)]
pub enum Color {
    Red,
    Green,
    Blue,
    Custom(u8, u8, u8),
}

/// A constant
pub const MAX_SIZE: usize = 1024;

/// A static
static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A type alias
pub type Result<T> = std::result::Result<T, MyError>;

pub(crate) fn crate_visible() {}
fn private_fn() {}
