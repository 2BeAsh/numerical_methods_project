using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SwordAttack : MonoBehaviour
{

    public float damage;
    Vector2 rightAttackOffset;
    public Collider2D swordCollider;

    private void Start()
    {
        rightAttackOffset = transform.position;
    }


    public void AttackRight()
    {
        swordCollider.enabled = true;
        transform.localPosition = rightAttackOffset;
    }


    public void AttackLeft()
    {
        swordCollider.enabled = true;
        transform.localPosition = new Vector2(rightAttackOffset.x * -1, rightAttackOffset.y);
    }


    public void StopAttack()
    {
        swordCollider.enabled = false;
    }

    private void OnTriggerEnter2D(Collider2D other)
    {
        if (other.tag == "Prey")
        { // Deal damage to prey
            Prey prey = other.GetComponent<Prey>();

            if (prey != null)
            {
                prey.Health -= damage;
            }
        }
    }


}
