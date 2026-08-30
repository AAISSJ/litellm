use std::sync::{Arc, Mutex};

use serde::Serialize;
use tracing::span::{Attributes, Id};
use tracing::{Dispatch, Level, Subscriber};
use tracing_subscriber::filter::{LevelFilter, filter_fn};
use tracing_subscriber::layer::Context;
use tracing_subscriber::prelude::*;
use tracing_subscriber::{Layer, Registry};

const TARGET: &str = "litellm::function_trace";

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct FunctionTraceEvent {
    pub(crate) function: &'static str,
}

#[derive(Clone, Default)]
pub(crate) struct FunctionTrace {
    events: Arc<Mutex<Vec<FunctionTraceEvent>>>,
}

impl FunctionTrace {
    pub(crate) fn dispatcher(&self) -> Dispatch {
        let filter = filter_fn(|metadata| {
            metadata.is_span() && metadata.target() == TARGET && *metadata.level() == Level::TRACE
        })
        .with_max_level_hint(LevelFilter::TRACE);
        Dispatch::new(
            Registry::default().with(
                FunctionTraceLayer {
                    trace: self.clone(),
                }
                .with_filter(filter),
            ),
        )
    }

    pub(crate) fn events(&self) -> Vec<FunctionTraceEvent> {
        self.events
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }
}

struct FunctionTraceLayer {
    trace: FunctionTrace,
}

impl<S> Layer<S> for FunctionTraceLayer
where
    S: Subscriber,
{
    fn on_new_span(&self, attributes: &Attributes<'_>, _id: &Id, _context: Context<'_, S>) {
        self.trace
            .events
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(FunctionTraceEvent {
                function: attributes.metadata().name(),
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_matching_spans_in_creation_order() {
        let trace = FunctionTrace::default();
        let dispatch = trace.dispatcher();

        tracing::dispatcher::with_default(&dispatch, || {
            let _ignored = tracing::trace_span!(target: "other", "ignored");
            let _first = tracing::trace_span!(target: TARGET, "same_name");
            let _wrong_level = tracing::debug_span!(target: TARGET, "wrong_level");
            let _second = tracing::trace_span!(target: TARGET, "same_name");
        });

        assert_eq!(
            trace.events(),
            vec![
                FunctionTraceEvent {
                    function: "same_name"
                },
                FunctionTraceEvent {
                    function: "same_name"
                },
            ]
        );
    }
}
