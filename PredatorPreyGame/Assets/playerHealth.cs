using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class playerHealth : MonoBehaviour

{
    public AudioSource audioPlayer_fireball_gotHit;
    public AudioSource you_dead;
    Animator animator;

    public int health = 10;
    private bool alive = true;

    private void Start()
    {
        animator = GetComponent<Animator>(); 
    }

    public void TakeDamage(int damage)
    {
        if (alive == true)
        {
            health -= damage;
            audioPlayer_fireball_gotHit.Play();
            if (health <= 0)
            {
                Die();
                alive = false;
            }
        }
    }

    void Die()
    {
        if (alive == true)
        {
            animator.SetTrigger("death");
            you_dead.Play();
        }
    }
}
