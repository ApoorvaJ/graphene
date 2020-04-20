use winit::event_loop::EventLoop;

fn main() {
    let event_loop = EventLoop::new();

    let ctx = grapheme::Context::new(&event_loop);
    ctx.main_loop(event_loop);
}
