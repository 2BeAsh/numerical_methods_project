using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class PlayerController : MonoBehaviour
{
    public float moveSpeed = 1f;
    public ContactFilter2D movementFilter;
    public float collisionOffset = 0.05f;
    public SwordAttack swordAttack;

    // Reference components added to Player object in Unity
    Vector2 movementInput;
    Rigidbody2D rb;
    List<RaycastHit2D> castCollisions = new List<RaycastHit2D>();
    Animator animator;
    SpriteRenderer spriteRenderer;

    // Start is called before the first frame update
    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
        animator = GetComponent<Animator>();
        spriteRenderer = GetComponent<SpriteRenderer>();
    }

    private void FixedUpdate()
    {
     // If movement is no 0, try to move
        if (movementInput != Vector2.zero){
            bool success = TryMove(movementInput);
            // If not succesfull in movement, try first only move in x direction
            // And if that is not succesfull, try only in y direction
            if (!success)
            {
                success = TryMove(new Vector2(movementInput.x, 0));

                if (!success)
                {
                    success = TryMove(new Vector2(0, movementInput.y));
                }
            }

            animator.SetBool("isMoving", success);
        } else
        {
            animator.SetBool("isMoving", false);
        }

        // Set moving direction of sprite relative to movement direction
        if (movementInput.x < 0)
        {
            spriteRenderer.flipX = true;
        } else if (movementInput.x > 0)
        {
            spriteRenderer.flipX = false;
        }

    }

    private bool TryMove(Vector2 direction)
    {
        if (direction == Vector2.zero) // If cannot move, then false. Prevents walking animation when moving into wall.
        {
            return false;
        }

        // Check for potential collision
        int count = rb.Cast(
            direction,
            movementFilter,
            castCollisions,
            moveSpeed * Time.fixedDeltaTime + collisionOffset);

        if (count == 0)
        { // Only move if number of collisions is 0
          // Position = pos + input * v * dt, input is factor of how much should move depending on player
            rb.MovePosition(rb.position + direction * moveSpeed * Time.fixedDeltaTime);
            return true;
        }
        else
        {
            return false;
        }

    }

    void OnMove(InputValue movementValue)
    {
        movementInput = movementValue.Get<Vector2>();
    }

    void OnFire()
    {
        animator.SetTrigger("swordAttack");
    }

    public void EndSwordAttack()
    {
        swordAttack.StopAttack();
    }

    public void SwordAttack()
    {
        if (spriteRenderer.flipX == true) // If are looking to the left, attack left
        {
            swordAttack.AttackLeft();
        } else
        {
            swordAttack.AttackRight(); // If are looking to the right, attack right
        }

    }






}
