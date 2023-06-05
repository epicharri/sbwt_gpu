#pragma once
#include "../globals.hpp"
#include "cuda.hpp"

namespace epic
{
    namespace gpu
    {

        struct DeviceStream
        {

            deviceStream_t stream;
            deviceError_t err;
            deviceEvent_t start;
            deviceEvent_t stop;
            deviceEvent_t start_2;
            deviceEvent_t stop_2;
            void create();
            void start_timer();
            void stop_timer();
            void start_timer_2();
            void stop_timer_2();
            float duration_in_millis();
            float duration_in_millis_2();
            void synchronize_stream();

            DeviceStream() = default;
            ~DeviceStream();
        };

        DeviceStream::~DeviceStream()
        {
            DEBUG_BEFORE_DESTRUCT("DeviceStream (ALL)");
            deviceEventDestroy(start);
            deviceEventDestroy(stop);
            deviceEventDestroy(start_2);
            deviceEventDestroy(stop_2);
            err = deviceStreamDestroy(stream);
            DEBUG_AFTER_DESTRUCT("DeviceStream (ALL)");
        }

        void DeviceStream::create()
        {
            err = deviceStreamCreate(&stream);
            deviceEventCreate(&start);
            deviceEventCreate(&stop);
            deviceEventCreate(&start_2);
            deviceEventCreate(&stop_2);
        }

        void DeviceStream::start_timer()
        {
            deviceEventRecord(start, stream);
        }

        void DeviceStream::stop_timer()
        {
            deviceEventRecord(stop, stream);
        }

        void DeviceStream::start_timer_2()
        {
            deviceEventRecord(start_2, stream);
        }

        void DeviceStream::stop_timer_2()
        {
            deviceEventRecord(stop_2, stream);
        }

        float DeviceStream::duration_in_millis()
        {
            float milliseconds;
            deviceEventSynchronize(stop);
            deviceEventElapsedTime(&milliseconds, start, stop);
            return milliseconds;
        }

        float DeviceStream::duration_in_millis_2()
        {
            float milliseconds;
            deviceEventSynchronize(stop_2);
            deviceEventElapsedTime(&milliseconds, start_2, stop_2);
            return milliseconds;
        }

        void DeviceStream::synchronize_stream()
        {
            deviceStreamSynchronize(stream);
        }
    }
}