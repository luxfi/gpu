// +build darwin,cgo

package gpu

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation
#include "metal/mtl_bridge.m"
*/
import "C"