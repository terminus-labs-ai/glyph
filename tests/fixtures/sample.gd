## A sample GDScript file for parser tests.
## Demonstrates class_name, extends, signals, enums, consts, vars, and funcs.
class_name PlayerController
extends CharacterBody2D

## Emitted when the player dies.
signal died()

## Emitted when health changes.
signal health_changed(old_value: int, new_value: int)

enum State {
	IDLE,
	RUNNING,
	JUMPING,
	DEAD,
}

const MAX_SPEED: float = 300.0
const JUMP_FORCE = -600

@export
var health: int = 100
var speed: float = 0.0
var state: State = State.IDLE

## Move the player based on input direction.
func move(direction: Vector2) -> void:
	velocity = direction * speed
	move_and_slide()

## Apply damage and emit health_changed.
func take_damage(amount: int) -> void:
	var old_health = health
	health -= amount
	health_changed.emit(old_health, health)
	if health <= 0:
		_die()

func _die() -> void:
	state = State.DEAD
	died.emit()

## Return the current speed as a float.
func get_speed() -> float:
	return speed
