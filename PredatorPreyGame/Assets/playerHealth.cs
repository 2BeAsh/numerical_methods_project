using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class playerHealth : MonoBehaviour

{
    public AudioSource audioPlayer_fireball_gotHit;
    public AudioSource you_dead;

    public int health = 10;
    Animator animator;


    private void Start()
    {
        animator = GetComponent<Animator>(); 
    }

    public void TakeDamage(int damage)
    {
        health -= damage;
        audioPlayer_fireball_gotHit.Play();
        if (health <= 0)
        {
            Die();
        }
    }

    void Die()
    {
        animator.SetTrigger("death");
        you_dead.Play();
    }
}
